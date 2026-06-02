from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    ids: list[str]
    documents: list[str]
    distances: list[float]
    metadatas: list[dict[str, Any]]
