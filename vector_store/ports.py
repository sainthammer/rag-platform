"""
Порт для взаимодействия с БД
"""

from abc import ABC, abstractmethod
from typing import Any

from vector_store.store_dataclasses import SearchResult


class VectorDB(ABC):
    """
    Абстрактный класс для реализации интерфейсов взаимодействия с БД
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    """
    Добавление записи/записей в коллекцию. Более подробно про аргументы в реализациях.
    """

    @abstractmethod
    def search(self, query_embedding: list[float], n_results: int = 3) -> SearchResult: ...

    """
    Поиск по эмбеддингу запроса, возвращает n релевантных записей(кол-во зависит от n_results)
    """

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    """
    Удаление записи/записей по id
    """

    @abstractmethod
    def count(self) -> int: ...

    """
    Кол-во записей в коллекции
    """
