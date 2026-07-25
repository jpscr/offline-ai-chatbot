"""Hybrid RAG query logic: if relevant local context is found, answer
strictly from it; otherwise fall back to the model's general knowledge."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config
from document_ingestion import DocumentIngestionPipeline
from llm_engine import LocalInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "You are a precise documentation assistant. Answer the user's question "
    "using ONLY the context provided below. If the context does not fully "
    "answer the question, say what is missing rather than guessing. Do not "
    "reference information outside the provided context.\n\n"
    "CONTEXT:\n{context}"
)

FALLBACK_SYSTEM_PROMPT = (
    "You are a knowledgeable general-purpose assistant. No relevant "
    "information was found in the user's local document collection for "
    "this question. Answer using your own general knowledge, and briefly "
    "note that this answer is not based on the user's local documents."
)


@dataclass
class RAGResponse:
    answer: str
    used_local_context: bool
    source_chunks: List[Document] = field(default_factory=list)


class HybridRAGEngine:
    """Orchestrates retrieval + generation with context-found/fallback branching."""

    def __init__(
        self,
        ingestion_pipeline: DocumentIngestionPipeline,
        inference_engine: LocalInferenceEngine,
        top_k: int = config.RETRIEVAL_TOP_K,
        distance_threshold: float = config.RELEVANCE_DISTANCE_THRESHOLD,
    ):
        self.ingestion_pipeline = ingestion_pipeline
        self.inference_engine = inference_engine
        self.top_k = top_k
        self.distance_threshold = distance_threshold

    def query(self, question: str) -> RAGResponse:
        vectorstore: Chroma = self.ingestion_pipeline.get_vectorstore()
        relevant_chunks = self._retrieve_relevant_chunks(vectorstore, question)

        if relevant_chunks:
            logger.info("Found %d relevant local chunk(s); answering from context.", len(relevant_chunks))
            context_text = "\n\n---\n\n".join(doc.page_content for doc in relevant_chunks)
            system_prompt = RAG_SYSTEM_PROMPT.format(context=context_text)
            answer = self.inference_engine.generate(system_prompt, question)
            return RAGResponse(answer=answer, used_local_context=True, source_chunks=relevant_chunks)

        logger.info("No sufficiently relevant local context found; falling back to world knowledge.")
        answer = self.inference_engine.generate(FALLBACK_SYSTEM_PROMPT, question)
        return RAGResponse(answer=answer, used_local_context=False, source_chunks=[])

    def _retrieve_relevant_chunks(self, vectorstore: Chroma, question: str) -> List[Document]:
        """Chroma distance is LOWER = closer match, so filter with `<=`."""
        results: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(
            question, k=self.top_k
        )
        for doc, distance in results:
            preview = doc.page_content[:60].replace("\n", " ")
            logger.info(
                "  retrieved distance=%.4f (threshold=%.4f) | %r...",
                distance, self.distance_threshold, preview,
            )
        return [doc for doc, distance in results if distance <= self.distance_threshold]