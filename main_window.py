"""Main application window: chat panel (75% width) + system telemetry
panel (25% width), wired to the background workers in workers.py."""

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from widgets import ChatBubble, MetricBar
from workers import ChatWorker, EngineBootWorker, IngestionWorker, SystemMonitorWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline Local AI & System Monitor Assistant")
        self.resize(1200, 760)
        self.setMinimumSize(900, 600)

        self.engine = None
        self.pipeline = None
        self._boot_worker = None
        self._chat_worker = None
        self._ingestion_worker = None
        self.monitor_worker = None

        self._build_ui()
        self._start_monitor()
        self._boot_engine()

    # UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_telemetry_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 300])
        root_layout.addWidget(splitter)

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("chatPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Local Assistant")
        title.setObjectName("chatTitle")
        header.addWidget(title)
        header.addStretch()
        self.status_label = QLabel("Starting engine\u2026")
        self.status_label.setObjectName("statusLabel")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.messages_area = QScrollArea()
        self.messages_area.setWidgetResizable(True)
        self.messages_area.setObjectName("messagesArea")
        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(4, 4, 4, 4)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch()
        self.messages_area.setWidget(self.messages_container)
        layout.addWidget(self.messages_area, stretch=1)

        input_row = QHBoxLayout()
        self.load_docs_button = QPushButton("Load Documents\u2026")
        self.load_docs_button.setObjectName("secondaryButton")
        self.load_docs_button.clicked.connect(self._on_load_documents_clicked)
        input_row.addWidget(self.load_docs_button)

        self.input_field = QLineEdit()
        self.input_field.setObjectName("chatInput")
        self.input_field.setPlaceholderText("Ask a question\u2026")
        self.input_field.returnPressed.connect(self._on_send_clicked)
        self.input_field.setEnabled(False)
        input_row.addWidget(self.input_field, stretch=1)

        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primaryButton")
        self.send_button.clicked.connect(self._on_send_clicked)
        self.send_button.setEnabled(False)
        input_row.addWidget(self.send_button)

        layout.addLayout(input_row)
        return panel

    def _build_telemetry_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("telemetryPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 20, 18, 20)
        layout.setSpacing(18)

        title = QLabel("System Telemetry")
        title.setObjectName("telemetryTitle")
        layout.addWidget(title)

        health_card = QFrame()
        health_card.setObjectName("telemetryCard")
        health_layout = QVBoxLayout(health_card)
        health_layout.setSpacing(12)
        health_header = QLabel("Host Health")
        health_header.setObjectName("cardHeader")
        health_layout.addWidget(health_header)

        self.cpu_bar = MetricBar("CPU")
        self.ram_bar = MetricBar("Memory")
        self.disk_bar = MetricBar("Disk")
        health_layout.addWidget(self.cpu_bar)
        health_layout.addWidget(self.ram_bar)
        health_layout.addWidget(self.disk_bar)
        layout.addWidget(health_card)

        modules_card = QFrame()
        modules_card.setObjectName("telemetryCard")
        modules_layout = QVBoxLayout(modules_card)
        modules_header = QLabel("Monitored Modules")
        modules_header.setObjectName("cardHeader")
        modules_layout.addWidget(modules_header)

        self.modules_table = QTableWidget(0, 4)
        self.modules_table.setObjectName("modulesTable")
        self.modules_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "RSS (MB)"])
        self.modules_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.modules_table.verticalHeader().setVisible(False)
        self.modules_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.modules_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.modules_table.setMaximumHeight(140)
        modules_layout.addWidget(self.modules_table)
        layout.addWidget(modules_card)

        service_card = QFrame()
        service_card.setObjectName("telemetryCard")
        service_layout = QHBoxLayout(service_card)
        service_label = QLabel("Ollama Service")
        service_label.setObjectName("cardHeader")
        service_layout.addWidget(service_label)
        service_layout.addStretch()
        self.ollama_status_dot = QLabel("\u25cf")
        self.ollama_status_dot.setObjectName("statusDotOffline")
        service_layout.addWidget(self.ollama_status_dot)
        self.ollama_status_label = QLabel("Checking\u2026")
        self.ollama_status_label.setObjectName("cardValue")
        service_layout.addWidget(self.ollama_status_label)
        layout.addWidget(service_card)

        layout.addStretch()
        return panel

    # Engine lifecycle -----------------------------------------------------

    def _boot_engine(self, docs_folder: Optional[str] = None) -> None:
        self._boot_worker = EngineBootWorker(docs_folder)
        self._boot_worker.ready.connect(self._on_engine_ready)
        self._boot_worker.failed.connect(self._on_engine_failed)
        self._boot_worker.start()

    def _on_engine_ready(self, engine) -> None:
        self.engine = engine
        self.pipeline = engine.ingestion_pipeline
        self._set_status("Ready", ready=True)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    def _on_engine_failed(self, message: str) -> None:
        self._set_status("Engine failed to start", ready=False)
        self._append_system_message(f"Startup error: {message}")

    def _set_status(self, text: str, ready: bool) -> None:
        self.status_label.setText(text)
        self.status_label.setObjectName("statusLabelReady" if ready else "statusLabel")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    # Chat -----------------------------------------------------------------

    def _on_send_clicked(self) -> None:
        question = self.input_field.text().strip()
        if not question or self.engine is None:
            return

        self._append_bubble(question, role="user")
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self._set_status("Thinking\u2026", ready=False)

        self._chat_worker = ChatWorker(self.engine, question)
        self._chat_worker.response_ready.connect(self._on_response_ready)
        self._chat_worker.failed.connect(self._on_chat_failed)
        self._chat_worker.start()

    def _on_response_ready(self, response) -> None:
        tag = "local docs" if response.used_local_context else "world knowledge"
        self._append_bubble(response.answer, role="assistant", source_tag=tag)
        self._set_status("Ready", ready=True)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    def _on_chat_failed(self, message: str) -> None:
        self._append_system_message(f"Error: {message}")
        self._set_status("Ready", ready=True)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)

    def _append_bubble(self, text: str, role: str, source_tag: Optional[str] = None) -> None:
        bubble = ChatBubble(text, role, source_tag)
        row = QHBoxLayout()
        if role == "user":
            row.addStretch()
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch()
        wrapper = QWidget()
        wrapper.setLayout(row)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, wrapper)
        self._scroll_to_bottom()

    def _append_system_message(self, text: str) -> None:
        label = QLabel(text)
        label.setObjectName("systemMessage")
        label.setWordWrap(True)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, label)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        bar = self.messages_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    # Document loading -------------------------------------------------------

    def _on_load_documents_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of .docx or .pdf files")
        if not folder:
            return
        if self.pipeline is None:
            self._append_system_message("Engine isn't ready yet -- try again in a moment.")
            return

        self.load_docs_button.setEnabled(False)
        self._set_status("Ingesting documents\u2026", ready=False)
        self._ingestion_worker = IngestionWorker(self.pipeline, folder)
        self._ingestion_worker.finished_ok.connect(self._on_ingestion_done)
        self._ingestion_worker.failed.connect(self._on_ingestion_failed)
        self._ingestion_worker.start()

    def _on_ingestion_done(self) -> None:
        self._set_status("Ready", ready=True)
        self.load_docs_button.setEnabled(True)
        self._append_system_message("Documents ingested. Ask away.")

    def _on_ingestion_failed(self, message: str) -> None:
        self._set_status("Ready", ready=True)
        self.load_docs_button.setEnabled(True)
        self._append_system_message(f"Ingestion error: {message}")

    # System telemetry -----------------------------------------------------

    def _start_monitor(self) -> None:
        self.monitor_worker = SystemMonitorWorker()
        self.monitor_worker.stats_updated.connect(self._on_stats_updated)
        self.monitor_worker.start()

    def _on_stats_updated(self, stats: dict) -> None:
        self.cpu_bar.set_value(stats["cpu_percent"])
        self.ram_bar.set_value(stats["ram_percent"])
        self.disk_bar.set_value(stats["disk_percent"])

        modules = stats["modules"]
        self.modules_table.setRowCount(len(modules))
        for row, module in enumerate(modules):
            self.modules_table.setItem(row, 0, QTableWidgetItem(str(module["pid"])))
            self.modules_table.setItem(row, 1, QTableWidgetItem(module["name"]))
            self.modules_table.setItem(row, 2, QTableWidgetItem(f'{module["cpu_percent"]:.1f}'))
            self.modules_table.setItem(row, 3, QTableWidgetItem(f'{module["rss_mb"]:.1f}'))

        online = stats["ollama_online"]
        self.ollama_status_dot.setObjectName("statusDotOnline" if online else "statusDotOffline")
        self.ollama_status_dot.style().unpolish(self.ollama_status_dot)
        self.ollama_status_dot.style().polish(self.ollama_status_dot)
        self.ollama_status_label.setText("Online" if online else "Unreachable")

    # Shutdown ---------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self.monitor_worker is not None:
            self.monitor_worker.stop()
            self.monitor_worker.wait(2000)
        for worker in (self._chat_worker, self._ingestion_worker, self._boot_worker):
            if worker is not None and worker.isRunning():
                worker.wait(2000)
        event.accept()
