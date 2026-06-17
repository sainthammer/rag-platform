"""Публичное API модуля embeddings."""

from .adapters import FakeEmbeddingService, OpenAIEmbeddingService, SentenceTransformersService
from .cache import EmbeddingCache, cached
from .ports import EmbeddingService
from .service import (
    CachedEmbeddingService,
    build_base_embedding_service,
    build_document_embedding_service,
    build_query_embedding_service,
)

__all__ = [
    "CachedEmbeddingService",
    "EmbeddingCache",
    "EmbeddingService",
    "FakeEmbeddingService",
    "OpenAIEmbeddingService",
    "SentenceTransformersService",
    "build_base_embedding_service",
    "build_document_embedding_service",
    "build_query_embedding_service",
    "cached",
]
