# Offline Local AI & System Monitor Assistant

A fully local, privacy-first desktop assistant that answers questions from
your own documents (.docx / .pdf) using retrieval-augmented generation
(RAG), with a live system telemetry panel alongside the chat. No API keys,
no cloud calls for inference, no data ever leaves the machine.

## Features

- **Hybrid RAG** — retrieves relevant chunks from your documents when
  available; falls back to the model's general knowledge otherwise, with
  every answer tagged so you know which source it came from.
- **Local inference via [Ollama](https://ollama.com)** — no cloud API,
  no API key, runs entirely on-device.
- **PyQt6 desktop GUI** — chat panel plus a live system telemetry sidebar
  (CPU/RAM/disk, watched processes, Ollama service health).
- **CLI mode** — a lightweight terminal interface for quick testing
  without launching the GUI.
- **Document formats** — ingests `.docx` and `.pdf` files from a folder
  you choose.

## Architecture

```
config.py              Central configuration (models, thresholds, paths)
document_ingestion.py  Parses .docx/.pdf -> chunks -> embeddings -> Chroma
llm_engine.py           HTTP client for a local Ollama server
rag_engine.py            Hybrid RAG query logic (context-found / fallback)
main.py                  CLI entry point
gui_app.py                GUI entry point
main_window.py             PyQt6 main window (chat + telemetry panel)
workers.py                  Background QThreads (boot, ingest, chat, monitor)
widgets.py                   Chat bubble + metric bar widgets
style.qss                     Application stylesheet
```

Document ingestion, embedding, and inference each run on background
`QThread`s so the GUI never freezes.

## Setup

### 1. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

Download from [ollama.com/download](https://ollama.com/download), then:

```bash
ollama pull llama3.2:3b
```

Use `llama3.2:3b` on 8GB RAM machines; a larger model (e.g. `llama3.1:8b`)
works on 16GB+ machines — just update `OLLAMA_MODEL_NAME` in `config.py`.

## Usage

**GUI:**
```bash
python gui_app.py
```

**CLI:**
```bash
python main.py --docs ./path/to/your/documents   # first run: ingest a folder
python main.py                                     # later runs: reuse the index
```

In the GUI, use the **Load Documents…** button to pick a folder of
`.docx`/`.pdf` files to ingest at any time.

**Try it immediately** with the bundled sample documents in `sample_docs/`:
```bash
python main.py --docs ./sample_docs
```
Then ask: *"What is the secret internal codename for this project?"* or
*"What is the designated hardware target for this deployment?"*

## Configuration

All tunables live in `config.py`:

| Setting | Purpose |
|---|---|
| `OLLAMA_MODEL_NAME` | Which local model to use |
| `LLM_CONTEXT_WINDOW` | Token budget per request (input + output) |
| `LLM_MAX_OUTPUT_TOKENS` | Max tokens generated per response |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Document chunking strategy |
| `RELEVANCE_DISTANCE_THRESHOLD` | Cutoff for "relevant enough" retrieval |

`RELEVANCE_DISTANCE_THRESHOLD` in particular is corpus-dependent — the
distances logged in the console on each query (`rag_engine.py`) are the
fastest way to recalibrate it for your own documents.

## Resource requirements

8–16GB RAM recommended. With a 4096-token context window, the model's KV
cache alone uses roughly 4.5–5.5GB — stick to `llama3.2:3b` on 8GB machines.
