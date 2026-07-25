"""Background QThread workers so the GUI thread never blocks on engine
startup, document ingestion, chat inference, or telemetry polling."""

import socket
import time
from typing import List, Optional

import psutil
from PyQt6.QtCore import QThread, pyqtSignal

import config
from document_ingestion import DocumentIngestionPipeline
from llm_engine import LocalInferenceEngine
from rag_engine import HybridRAGEngine


class EngineBootWorker(QThread):
    """Builds the ingestion pipeline, inference engine, and RAG engine once
    at startup (loads the embedding model, verifies Ollama is reachable)."""

    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, docs_folder: Optional[str] = None):
        super().__init__()
        self.docs_folder = docs_folder

    def run(self) -> None:
        try:
            pipeline = DocumentIngestionPipeline()
            if self.docs_folder:
                pipeline.ingest_folder(self.docs_folder)
            else:
                pipeline.load_existing_store()

            inference = LocalInferenceEngine()
            engine = HybridRAGEngine(pipeline, inference)
            self.ready.emit(engine)
        except Exception as exc:
            self.failed.emit(str(exc))


class IngestionWorker(QThread):
    """Re-runs ingestion against a newly chosen folder without blocking the UI."""

    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, pipeline: DocumentIngestionPipeline, docs_folder: str):
        super().__init__()
        self.pipeline = pipeline
        self.docs_folder = docs_folder

    def run(self) -> None:
        try:
            self.pipeline.ingest_folder(self.docs_folder)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class ChatWorker(QThread):
    """Runs one query() call off the GUI thread."""

    response_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, engine: HybridRAGEngine, question: str):
        super().__init__()
        self.engine = engine
        self.question = question

    def run(self) -> None:
        try:
            response = self.engine.query(self.question)
            self.response_ready.emit(response)
        except Exception as exc:
            self.failed.emit(str(exc))


class SystemMonitorWorker(QThread):
    """Polls host CPU/RAM/disk, watched processes, and Ollama reachability
    on a fixed interval, emitting one signal per tick."""

    stats_updated = pyqtSignal(dict)

    def __init__(
        self,
        watched_names: Optional[List[str]] = None,
        poll_interval_ms: int = config.SYSTEM_MONITOR_POLL_INTERVAL_MS,
    ):
        super().__init__()
        self.watched_names = [n.lower() for n in (watched_names or config.WATCHED_PROCESS_NAMES)]
        self.poll_interval_s = poll_interval_ms / 1000.0
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the sample window
        while self._running:
            stats = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "modules": self._scan_watched_processes(),
                "ollama_online": self._check_ollama_reachable(),
            }
            self.stats_updated.emit(stats)
            slept = 0.0
            while slept < self.poll_interval_s and self._running:
                time.sleep(0.2)
                slept += 0.2

    def _scan_watched_processes(self) -> List[dict]:
        modules = []
        for proc in psutil.process_iter(["pid", "name", "status"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if not any(watched in name for watched in self.watched_names):
                    continue
                cpu = proc.cpu_percent(interval=0.1)  # brief blocking sample, matches only
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                modules.append({
                    "pid": proc.info["pid"],
                    "name": proc.info.get("name") or "?",
                    "status": proc.info.get("status") or "?",
                    "cpu_percent": cpu,
                    "rss_mb": rss_mb,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return modules

    def _check_ollama_reachable(self) -> bool:
        """TCP connect check to the actual local service this app depends
        on, avoiding OS-specific `ping` flag differences."""
        try:
            with socket.create_connection(
                (config.OLLAMA_HEALTH_HOST, config.OLLAMA_HEALTH_PORT), timeout=1.0
            ):
                return True
        except OSError:
            return False
