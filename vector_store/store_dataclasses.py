"""
Базовые структуры для работы с БД
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_OP_MAP = {
    "eq": "$eq",
    "ne": "$ne",
    "gt": "$gt",
    "gte": "$gte",
    "lt": "$lt",
    "lte": "$lte",
    "in": "$in",
    "nin": "$nin",
}


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


@dataclass
class MetadataScheme:
    """
    Метаданные чанка

    args:
        source_file: путь или имя сорс файла
        page_num: номер страницы???
        chunk_id: номер чанка???
        language: язык текста
        created_at: когда чанк был создан/проиндексирован
        section: часть(раздел) документа
    """

    source_file: str
    page_num: int
    chunk_id: int
    language: str
    created_at: datetime = field(default_factory=datetime.now)
    section: str = ""

    def to_dict(self):
        """
        Сериализация для передачи в БД как аргумент metadatas
        """
        return {
            "source_file": self.source_file,
            "page_num": self.page_num,
            "chunk_id": self.chunk_id,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "section": self.section,
        }


@dataclass
class MetadataFilter:
    """
    Универсальный фильтр по метаданным

    args:
        field: название поля из MetadataScheme ("language", "page_num", ...)
        op: оператор сравнения из _OP_MAP
        value: значение для сравнения ("ru", 3, ["ru", "en"], ...)
    """

    field: str
    op: str
    value: Any

    def to_chroma_where(self) -> dict:
        """
        Передает фильтр в Chroma where-clause
        """

        chroma_op = _OP_MAP.get(self.op)
        if chroma_op is None:
            raise ValueError(f"Unknown operator: {self.op}. Default operators: {list(_OP_MAP)}")

        return {self.field: {chroma_op: self.value}}


def combine_filters(filters: list[MetadataFilter]) -> dict | None:
    """
    Объединяет несколько фильтров через $and
    Возвращает None если список пустой
    """

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0].to_chroma_where()
    return {"$and": [f.to_chroma_where() for f in filters]}
