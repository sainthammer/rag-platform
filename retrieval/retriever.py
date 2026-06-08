"""Retriever: векторный поиск с опциональным реранкингом.

Поток данных retrieve():
    query (str)
      │
      ├─▶ embed_fn(query)                       → вектор запроса
      ├─▶ vector_db_factory(collection).search() → кандидаты (список SourceChunk)
      ├─▶ post-search фильтрация по metadata    (если filters задан)
      ├─▶ reranker.rerank(query, candidates)    (если reranker задан)
      └─▶ candidates[:top_k]                    → List[SourceChunk]
"""

from __future__ import annotations

from typing import Any, Callable

from vector_store.ports import VectorDB

from .pipeline import SourceChunk, _distance_to_score
from .ports import Reranker


class Retriever:
    """Компонент поиска: от текстового запроса до ранжированного списка чанков.

    Отделён от RAGPipeline: не выполняет генерацию текста, только поиск.
    Это позволяет использовать его самостоятельно (например, для оценки
    качества retrieval или в MCP-инструментах).

    Args:
        embed_fn: Функция ``text → vector`` для векторизации запроса.
        vector_db_factory: Фабрика ``collection_name → VectorDB``.
        reranker: Реранкер результатов поиска. Если ``None`` — возвращаются
            результаты в порядке расстояния от векторного поиска.
        fetch_k: Сколько кандидатов запрашивать из векторного поиска перед
            реранкингом. Если ``None`` — вычисляется как ``top_k * 3``
            (для реранкера) или ``top_k`` (без реранкера).
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        vector_db_factory: Callable[[str], VectorDB],
        reranker: Reranker | None = None,
        fetch_k: int | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.vector_db_factory = vector_db_factory
        self.reranker = reranker
        self.fetch_k = fetch_k

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SourceChunk]:
        """Найти наиболее релевантные чанки для запроса.

        Args:
            query: Текстовый запрос пользователя.
            collection: Имя коллекции в векторном хранилище.
            top_k: Количество результатов в итоговом ответе.
            filters: Словарь для фильтрации по metadata чанков.
                Применяется после векторного поиска (post-filter).
                Пример: ``{"source": "doc.txt"}`` — вернуть только чанки
                из файла ``doc.txt``.

        Returns:
            Список ``SourceChunk`` длиной не более ``top_k``, отсортированный
            по убыванию score. Список может быть короче ``top_k``, если
            коллекция содержит меньше документов или после фильтрации
            осталось мало кандидатов.
        """
        # Определяем, сколько запросить из БД
        if self.fetch_k is not None:
            n_fetch = self.fetch_k
        elif self.reranker is not None:
            n_fetch = top_k * 3  # Берём больше кандидатов для реранкинга
        else:
            n_fetch = top_k

        # Векторизация запроса и поиск в БД
        query_vector = self.embed_fn(query)
        db = self.vector_db_factory(collection)
        raw = db.search(query_vector, n_results=n_fetch)

        # Преобразуем в SourceChunk
        candidates: list[SourceChunk] = [
            SourceChunk(
                text=doc,
                score=_distance_to_score(dist),
                doc_id=doc_id,
                metadata=meta,
            )
            for doc_id, doc, dist, meta in zip(
                raw.ids,
                raw.documents or [],
                raw.distances,
                raw.metadatas or [],
            )
        ]

        # Post-search фильтрация по metadata
        if filters:
            candidates = [
                c for c in candidates
                if all(c.metadata.get(k) == v for k, v in filters.items())
            ]

        # Реранкинг
        if self.reranker is not None and candidates:
            candidates = self.reranker.rerank(query, candidates)

        return candidates[:top_k]
