"""Пакет embeddings: сервис векторизации текстов.

Экспортирует:
    EmbeddingService — получение эмбеддингов через OpenAI API
                       с OTel-трейсингом и Prometheus-метриками.
"""

from embeddings.service import EmbeddingService

__all__ = ["EmbeddingService"]
