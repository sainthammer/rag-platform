from abc import ABC, abstractmethod
from typing import Any

from .store_dataclasses import SearchResult


class VectorDB(ABC):
    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], n_results: int = 3) -> SearchResult: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...
