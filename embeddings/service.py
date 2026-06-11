"""Композиция embedding-сервисов."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapters import FakeEmbeddingService, OllamaEmbeddingService, OpenAIEmbeddingService, SentenceTransformersService
from .cache import EmbeddingCache, cached
from .ports import EmbeddingService

if TYPE_CHECKING:
    from config import Settings


class CachedEmbeddingService(EmbeddingService):
    """Обёртка над ``EmbeddingService`` с SQLite-кэшированием.

    Сначала пытается получить вектор из ``EmbeddingCache``. Если значения
    нет, вызывает базовый сервис, сохраняет результат и возвращает его.
    """

    def __init__(
        self,
        base: EmbeddingService,
        cache: EmbeddingCache,
        model_name: str,
    ) -> None:
        """Сохранить базовый сервис, кэш и имя модели для ключей кэша.

        Args:
            base: Реальный embedding-сервис, который считает отсутствующие векторы.
            cache: SQLite-кэш embedding-векторов.
            model_name: Имя модели для построения ключей кэша.
        """
        self.base = base
        self.cache = cache
        self.model_name = model_name

    @cached
    def embed(self, text: str) -> list[float]:
        """Вернуть embedding одного текста с использованием кэша.

        Args:
            text: Текст для векторизации.

        Returns:
            Embedding-вектор текста.
        """
        return self.base.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Вернуть embeddings батча, считая только отсутствующие в кэше тексты.

        Args:
            texts: Список текстов для векторизации.

        Returns:
            Список embedding-векторов в порядке входных текстов.
        """
        if not texts:
            return []

        cached_vectors = self.cache.get_many(texts, self.model_name)
        result: list[list[float] | None] = list(cached_vectors)

        missing_indexes = [
            index for index, vector in enumerate(cached_vectors) if vector is None
        ]
        if missing_indexes:
            missing_texts = [texts[index] for index in missing_indexes]
            missing_vectors = self.base.embed_batch(missing_texts)
            self.cache.set_many(missing_texts, self.model_name, missing_vectors)
            for index, vector in zip(missing_indexes, missing_vectors, strict=True):
                result[index] = vector

        assert all(v is not None for v in result), (
            "embed_batch: не удалось вычислить векторы для всех входных текстов"
        )
        return result  # type: ignore[return-value]

    def dimension(self) -> int:
        """Вернуть размерность базового embedding-сервиса.

        Returns:
            Количество координат embedding-вектора.
        """
        return self.base.dimension()


def build_embedding_service(s: Settings | None = None) -> EmbeddingService:
    """Создать embedding-сервис по настройкам приложения.

    Фабрика читает ``EMBEDDING_PROVIDER``, ``EMBEDDING_MODEL`` и параметры
    кэша. Возвращает либо провайдер, либо ``CachedEmbeddingService``.

    Args:
        s: Настройки приложения. Если ``None``, используется глобальный ``settings``.

    Returns:
        Готовый embedding-сервис.

    Raises:
        ValueError: Если указан неизвестный embedding-провайдер.
    """
    if s is None:
        from config import settings

        s = settings

    if s.embedding_provider == "openai":
        base: EmbeddingService = OpenAIEmbeddingService(
            model=s.embedding_model,
            api_key=s.openai_api_key or None,
            normalize=s.embedding_normalize,
        )
    elif s.embedding_provider == "sentence-transformers":
        base = SentenceTransformersService(
            model_name=s.embedding_model,
            normalize=s.embedding_normalize,
        )
    elif s.embedding_provider == "ollama":
        base = OllamaEmbeddingService(
            model=s.embedding_model,
            base_url=s.ollama_base_url,
            normalize=s.embedding_normalize,
        )
    elif s.embedding_provider == "fake":
        base = FakeEmbeddingService(size=s.embedding_dimension)
    else:
        raise ValueError(f"Неизвестный эмбеддинг провайдер: {s.embedding_provider!r}")

    if not s.embedding_cache_enabled:
        return base

    return CachedEmbeddingService(
        base=base,
        cache=EmbeddingCache(s.embedding_cache_path),
        model_name=s.embedding_model,
    )
