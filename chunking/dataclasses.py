"""Общие структуры данных для загрузки документов и чанкинга."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """Текстовый фрагмент, готовый к векторизации и записи в vector store.

    Args:
        text: Текст чанка.
        metadata: Метаданные источника и параметры разбиения.
        id: Стабильный идентификатор чанка.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""
