"""Central configuration for the Offline Local AI & System Monitor Assistant."""

from pathlib import Path

# Filesystem paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# Local inference backend: Ollama (https://ollama.com/download/mac)
# Pull a model with: ollama pull llama3.2:3b
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "llama3.2:3b"  # 3B recommended for 8GB RAM machines

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking strategy: 1000 chars keeps chunks topically coherent while
# leaving room in the context window for retrieved chunks + chat history.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval tuning
RETRIEVAL_TOP_K = 4

# Chroma distance is LOWER = more similar. Above this threshold, the
# hybrid engine falls back to the model's general knowledge instead of
# using local documents. Recalibrate against your own corpus if needed.
RELEVANCE_DISTANCE_THRESHOLD = 1.5

# LLM inference parameters
LLM_CONTEXT_WINDOW = 4096
LLM_MAX_OUTPUT_TOKENS = 512
LLM_TEMPERATURE = 0.3

# GUI / system monitor settings
WATCHED_PROCESS_NAMES = ["python", "ollama"]
SYSTEM_MONITOR_POLL_INTERVAL_MS = 3000
METRIC_WARNING_THRESHOLD = 60
METRIC_CRITICAL_THRESHOLD = 85

# Health check for the local Ollama service (used by the telemetry panel
# instead of pinging an external IP, since Ollama is what this app depends on)
OLLAMA_HEALTH_HOST = "127.0.0.1"
OLLAMA_HEALTH_PORT = 11434
