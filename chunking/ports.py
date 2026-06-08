"""Абстрактные типы модуля chunking: Chunk, Chunker, DocumentLoader.

Все конкретные реализации зависят только от этих контрактов,
остальной код должен зависеть от интерфейсов, а не от конкретики.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    """Текстовый фрагмент документа с метаданными и идентификатором.

    Attributes:
        text: Текстовое содержимое фрагмента.
        metadata: Произвольные метаданные (источник, индекс, заголовок и т.п.).
        id: Уникальный идентификатор чанка внутри документа.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""


class Chunker(ABC):
    """Базовый класс для стратегий разбивки текста на фрагменты."""

    @abstractmethod
    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Разбить текст на список чанков.

        Args:
            text: Исходный текст для разбивки.
            metadata: Базовые метаданные, копируемые в каждый чанк.

        Returns:
            Список Chunk-объектов в порядке появления в тексте.
        """
        ...


class DocumentLoader(ABC):
    """Базовый класс для загрузчиков документов разных форматов."""

    @abstractmethod
    def load(self, path: str | Path) -> str:
        """Загрузить документ и вернуть его текстовое содержимое.

        Args:
            path: Путь к файлу документа.

        Returns:
            Извлечённый текст документа.
        """
        ...
