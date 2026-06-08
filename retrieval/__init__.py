"""Публичное API модуля retrieval."""

from .pipeline import FALLBACK_ANSWER, RAGPipeline, RAGResponse, SourceChunk

__all__ = [
    "FALLBACK_ANSWER",
    "RAGPipeline",
    "RAGResponse",
    "SourceChunk",
]
