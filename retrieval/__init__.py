"""Публичное API модуля retrieval."""

from .pipeline import FALLBACK_ANSWER, RAGPipeline, RAGResponse, SourceChunk, build_rag_pipeline

__all__ = [
    "FALLBACK_ANSWER",
    "RAGPipeline",
    "RAGResponse",
    "SourceChunk",
    "build_rag_pipeline",
]
