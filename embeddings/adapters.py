"""Конкретные реализации ``EmbeddingService``.

Модуль содержит адаптеры к внешним embedding-провайдерам и тестовую
детерминированную реализацию. Все классы следуют контракту из
``embeddings.ports``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .ports import EmbeddingService
from .utils import normalize_batch

OPENAI_BATCH_LIMIT = 2048

_OPENAI_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingService(EmbeddingService):
    """Сервис эмбеддингов через OpenAI Embeddings API.

    Args:
        model: Имя embedding-модели OpenAI.
        api_key: API-ключ. Если ``None``, SDK прочитает ключ из окружения.
        normalize: Нужно ли применять L2-нормализацию.
        batch_limit: Максимальный размер батча. По заданию это 2048 текстов.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        normalize: bool = True,
        batch_limit: int = OPENAI_BATCH_LIMIT,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.normalize = normalize
        self.batch_limit = batch_limit
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in _batched(texts, self.batch_limit):
            response = self.client.embeddings.create(model=self.model, input=list(batch))
            batch_vectors = [item.embedding for item in response.data]
            vectors.extend(batch_vectors)

        return normalize_batch(vectors) if self.normalize else vectors

    def dimension(self) -> int:
        if self.model in _OPENAI_DIMENSIONS:
            return _OPENAI_DIMENSIONS[self.model]
        return len(self.embed("Текст для измерения длины эмбеддинга"))


class SentenceTransformersService(EmbeddingService):
    """Сервис эмбеддингов на базе SentenceTransformers.

    Args:
        model_name: Имя модели, например ``BAAI/bge-m3``.
        normalize: Нужно ли нормализовать эмбеддинги через ``encode``.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        encoded = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=False,
        )
        return [_as_float_list(vector) for vector in encoded]

    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()

        if dimension is not None:
            return int(dimension)
        return len(self.embed("Текст для измерения длины эмбеддинга"))


class FakeEmbeddingService(EmbeddingService):
    """Детерминированный сервис эмбеддингов для unit-тестов.

    Не использует внешние зависимости и сеть. Один и тот же текст всегда
    даёт один и тот же вектор, поэтому класс удобен для проверки кэша.
    """

    def __init__(
        self,
        size: int = 8,
        model_name: str = "fake-embedding",
        normalize: bool = False,
    ) -> None:
        self.size = size
        self.model_name = model_name
        self.normalize = normalize
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = [self._embed_one(text) for text in texts]
        return normalize_batch(vectors) if self.normalize else vectors

    def dimension(self) -> int:
        return self.size

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        values = [digest[i % len(digest)] / 255.0 for i in range(self.size)]
        if all(value == 0 for value in values):
            values[0] = 1.0
        return values


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("Размер батча должен быть больше ноля")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _as_float_list(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    if isinstance(vector, list):
        return [float(value) for value in vector]
    return [float(value) for value in vector]  # type: ignore[union-attr]
