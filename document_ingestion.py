"""Document ingestion pipeline: parses .docx/.pdf files, chunks them,
embeds them locally, and persists the vectors to a Chroma database."""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Owns the lifecycle of turning raw .docx/.pdf files into a queryable,
    persisted vector store."""

    def __init__(
        self,
        persist_directory: str = config.CHROMA_PERSIST_DIR,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info("Loading embedding model '%s'...", embedding_model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self._vectorstore: Optional[Chroma] = None

    def ingest_folder(self, source_folder: str) -> Chroma:
        """Parse every .docx/.pdf in source_folder, chunk, embed, and persist."""
        source_path = Path(source_folder)
        if not source_path.exists():
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

        logger.info("Scanning '%s' for .docx and .pdf files...", source_folder)
        docx_loader = DirectoryLoader(
            source_folder, glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True
        )
        pdf_loader = DirectoryLoader(
            source_folder, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        raw_documents = docx_loader.load() + pdf_loader.load()

        if not raw_documents:
            logger.warning("No .docx or .pdf files found in %s", source_folder)
            return self.load_existing_store()

        logger.info("Loaded %d document(s). Splitting into chunks...", len(raw_documents))
        chunks = self._split_documents(raw_documents)
        logger.info(
            "Produced %d chunk(s). Embedding + persisting to '%s'...",
            len(chunks), self.persist_directory,
        )

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        if hasattr(self._vectorstore, "persist"):
            self._vectorstore.persist()

        logger.info("Ingestion complete.")
        return self._vectorstore

    def load_existing_store(self) -> Chroma:
        """Load a previously persisted Chroma database without re-embedding."""
        if self._vectorstore is None:
            logger.info("Loading persisted vector store from '%s'...", self.persist_directory)
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        return self._vectorstore

    def get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            return self.load_existing_store()
        return self._vectorstore

    def _split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(documents)