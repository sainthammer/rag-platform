"""Публичное API модуля retrieval."""

from .pipeline import FALLBACK_ANSWER, RAGPipeline, RAGResponse, SourceChunk, build_rag_pipeline
from .ports import Reranker
from .rerankers import CrossEncoderReranker, MMRReranker
from .retriever import Retriever

__all__ = [
    "FALLBACK_ANSWER",
    "RAGPipeline",
    "RAGResponse",
    "SourceChunk",
    "build_rag_pipeline",
    "Reranker",
    "CrossEncoderReranker",
    "MMRReranker",
    "Retriever",
]
