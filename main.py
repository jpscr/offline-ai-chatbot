"""CLI entry point for the RAG backend.

Usage:
    python main.py --docs ./my_docs      # first run: ingest a folder
    python main.py                       # later runs: load persisted index

For the GUI instead, run gui_app.py.
"""

import argparse
import sys

from document_ingestion import DocumentIngestionPipeline
from llm_engine import LocalInferenceEngine
from rag_engine import HybridRAGEngine


def build_engine(docs_folder: str) -> HybridRAGEngine:
    ingestion = DocumentIngestionPipeline()

    if docs_folder:
        ingestion.ingest_folder(docs_folder)
    else:
        ingestion.load_existing_store()

    inference = LocalInferenceEngine()
    return HybridRAGEngine(ingestion, inference)


def main():
    parser = argparse.ArgumentParser(description="Offline Local AI Assistant - CLI")
    parser.add_argument(
        "--docs",
        default=None,
        help="Path to a folder of .docx or .pdf files to ingest before starting the chat loop.",
    )
    args = parser.parse_args()

    try:
        engine = build_engine(args.docs)
    except FileNotFoundError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Ready. Type a question (or 'exit' to quit).")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        response = engine.query(question)
        source_tag = "[local docs]" if response.used_local_context else "[world knowledge]"
        print(f"\n{source_tag} {response.answer}")


if __name__ == "__main__":
    main()