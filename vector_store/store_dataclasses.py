"""
Базовые структуры для работы с БД
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """
    Структура возвращаемого ответа при поиске

    количество возвращенных записей зависит от n_results

    attrs:
        ids: список id найденых записей
        documents: список документов для найденых записей
        distances: схожесть эмбеддингов
        metadatas: связанные с записями словари с метаданными
    """

    ids: list[str]
    documents: list[str]
    distances: list[float]
    metadatas: list[dict[str, Any]]
