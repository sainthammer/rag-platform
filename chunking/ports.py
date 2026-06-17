
"""Абстрактные интерфейсы загрузчиков документов и chunker-стратегий."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .dataclasses import Chunk


class DocumentLoader(ABC):
    """Читает один исходный файл и возвращает текст с метаданными."""

    @abstractmethod
    def load(self, source_path: str | Path) -> tuple[str, dict[str, object]]:
        """Загрузить документ с диска.

        Args:
            source_path: Путь к исходному документу.

        Returns:
            Кортеж из нормализованного текста и метаданных источника.

        Raises:
            OSError: Если файл нельзя прочитать.
        """
        ...


class Chunker(ABC):
    """Разбивает загруженный текст на стабильные чанки."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, object] | None = None) -> list[Chunk]:
        """Разбить текст на чанки.

        Args:
            text: Исходный текст документа.
            metadata: Метаданные, которые нужно унаследовать каждому чанку.

        Returns:
            Список чанков с текстом, метаданными и стабильными id.

        Raises:
            ValueError: Если параметры конкретной стратегии некорректны.
        """
        ...
