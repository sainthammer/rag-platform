"""Публичное API модуля embeddings."""

from .adapters import FakeEmbeddingService, OpenAIEmbeddingService, SentenceTransformersService
from .cache import EmbeddingCache, cached
from .ports import EmbeddingService
from .service import CachedEmbeddingService, build_embedding_service

__all__ = [
    "CachedEmbeddingService",
    "EmbeddingCache",
    "EmbeddingService",
    "FakeEmbeddingService",
    "OpenAIEmbeddingService",
    "SentenceTransformersService",
    "build_embedding_service",
    "cached",
]
