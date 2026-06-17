"""Композиция embedding-сервисов."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapters import OpenAIEmbeddingService, SentenceTransformersService
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

        missing_indexes_by_text: dict[str, list[int]] = {}
        missing_texts: list[str] = []
        for index, vector in enumerate(cached_vectors):
            if vector is not None:
                continue

            text = texts[index]
            if text not in missing_indexes_by_text:
                missing_indexes_by_text[text] = []
                missing_texts.append(text)
            missing_indexes_by_text[text].append(index)

        if missing_texts:
            missing_vectors = self.base.embed_batch(missing_texts)
            self.cache.set_many(missing_texts, self.model_name, missing_vectors)
            for text, vector in zip(missing_texts, missing_vectors, strict=True):
                for index in missing_indexes_by_text[text]:
                    result[index] = vector

        completed: list[list[float]] = []
        for vector in result:
            if vector is None:
                raise RuntimeError("Не удалось заполнить embedding-вектор после cache miss")
            completed.append(vector)
        return completed

    def dimension(self) -> int:
        """Вернуть размерность базового embedding-сервиса.

        Returns:
            Количество координат embedding-вектора.
        """
        return self.base.dimension()


def build_base_embedding_service(s: Settings | None = None) -> EmbeddingService:
    """Создать базовый embedding-сервис без SQLite-кэша.

    Фабрика читает настройки embedding-провайдера, модели и нормализации.
    Результат можно использовать напрямую для чанков документов или обернуть
    в ``CachedEmbeddingService`` для пользовательских запросов.

    Args:
        s: Настройки приложения. Если ``None``, используется глобальный
            объект ``settings`` из ``config.py``.

    Returns:
        Базовый embedding-провайдер без ``CachedEmbeddingService``.

    Raises:
        ValueError: Если указан неизвестный embedding-провайдер.
    """
    if s is None:
        from config import settings

        s = settings

    if s.embedding_provider == "openai":
        return OpenAIEmbeddingService(
            model=s.embedding_model,
            api_key=s.openai_api_key or None,
            normalize=s.embedding_normalize,
        )
    if s.embedding_provider == "sentence-transformers":
        return SentenceTransformersService(
            model_name=s.embedding_model,
            normalize=s.embedding_normalize,
        )

    raise ValueError(f"Неизвестный embedding-провайдер: {s.embedding_provider!r}")


def build_query_embedding_service(s: Settings | None = None) -> EmbeddingService:
    """Создать embedding-сервис для пользовательских запросов.

    Фабрика читает настройки embedding-провайдера и, если включен
    ``EMBEDDING_CACHE_ENABLED``, оборачивает базовый провайдер в
    ``CachedEmbeddingService``. Этот путь предназначен для query embeddings,
    где повторные запросы пользователя имеет смысл переиспользовать из кэша.

    Args:
        s: Настройки приложения. Если ``None``, используется глобальный
            объект ``settings`` из ``config.py``.

    Returns:
        Готовый embedding-сервис для пользовательских запросов. При включенном
        кэше возвращает ``CachedEmbeddingService``, иначе базовый провайдер.

    Raises:
        ValueError: Если указан неизвестный embedding-провайдер.
    """
    if s is None:
        from config import settings

        s = settings

    base = build_base_embedding_service(s)

    if not s.embedding_cache_enabled:
        return base

    return CachedEmbeddingService(
        base=base,
        cache=EmbeddingCache(s.embedding_cache_path),
        model_name=s.embedding_model,
    )


def build_document_embedding_service(s: Settings | None = None) -> EmbeddingService:
    """Создать embedding-сервис для чанков документов без SQLite-кэша.

    Embedding-и чанков должны сохраняться в vector store вместе с id,
    документами и метаданными. Поэтому эта фабрика всегда возвращает базовый
    embedding-провайдер и игнорирует ``EMBEDDING_CACHE_ENABLED``.

    Args:
        s: Настройки приложения. Если ``None``, используется глобальный
            объект ``settings`` из ``config.py``.

    Returns:
        Базовый embedding-провайдер без ``CachedEmbeddingService``.

    Raises:
        ValueError: Если указан неизвестный embedding-провайдер.
    """
    return build_base_embedding_service(s)
