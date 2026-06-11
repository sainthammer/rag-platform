"""
Реализации интерфейсов для базового взаимодействия с векторными БД.
"""

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from vector_store.bm25 import BM25SparseVectorizer
from vector_store.store_dataclasses import CollectionStats, SearchResult
from vector_store.utils import to_uuid

from .ports import VectorDB

_RETRY = dict(
    retry=retry_if_exception_type((ConnectionError, OSError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

# ----------------- ChromaDB -----------------


class ChromaDB(VectorDB):
    """
    Реализация для ChromaDB

    args:
        collection: название создоваемой/существующей коллекции
        persists_directory: директория, в котрой будет хранится файл БД
        host: имя хоста, где запущена БД при сетевом деплое
        port: порт, по которому доступна БД при сетевом деплое
    """

    def __init__(
        self,
        collection: str,
        persist_directory: str | None = None,
        host: str = "localhost",
        port: int = 8000,
    ) -> None:
        import chromadb

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.HttpClient(host=host, port=port)

        self.collection = self.client.get_or_create_collection(collection)

    @retry(**_RETRY)
    def add(self, ids, embeddings, documents=None, metadatas=None) -> None:
        """
        Добавление записей в коллекцию.
        Все передаваемые списки должны быть одинаковой длинны, иначе обрежет на самом коротком

        args:
            ids: список идентификаторов файлов ??? пока хз, надо обсудить формат передаваемых данных
            emdeddings: список эмбеддингов, соответсвующих документам
            documents: список самих документов
            metadatas: список словарей с метаданными для документов(теги и тп)

        """
        self.collection.upsert(
            ids=[to_uuid(id) for id in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    @retry(**_RETRY)
    def search(self, query_embedding, n_results=3) -> SearchResult:
        """
        Функция поиска по коллекции

        args:
            qury_embedding: эмбеддинг запроса, по которому делаем поиск
            n_results: количество ближайших/релевантных записей в возвращенном результате
            include: какие поля будет включать возвращенные записи
        """
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return SearchResult(
            ids=raw["ids"][0],
            documents=(raw["documents"] or [[]])[0],
            distances=(raw["distances"] or [[]])[0],
            metadatas=[dict(m) if m is not None else {} for m in (raw["metadatas"] or [[]])[0]],
        )

    @retry(**_RETRY)
    def delete(self, ids) -> None:
        """
        удаление записи/записей по ее/их id

        args:
            ids: cписок id на удаление
        """
        self.collection.delete(ids=[to_uuid(id) for id in ids])

    @retry(**_RETRY)
    def count(self) -> int:
        """
        Возвращает количество записей в коллекции
        """
        return self.collection.count()

    @retry(**_RETRY)
    def get_stats(self) -> CollectionStats:
        return CollectionStats(
            collection=self.collection.name,
            vectors_count=self.collection.count(),
        )


# ----------------- Qdrant -----------------


class QdrantDB(VectorDB):
    """
    Реализация интрфейса для Qdrant

    args:
        collection: название коллекции, которую создаем
        vector_size: n-мерность векторов в коллекции
        host: имя хоста, где задеплоена БД
        port: порт, по которому доступна БД на хосте
        in_memory: флаг, который позволяет запустить БД в ОП
    """

    def __init__(
        self,
        collection: str,
        vector_size: int,
        host: str = "localhost",
        port: int = 6333,
        in_memory: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)

        self.collection = collection

        existing_collections = [
            collection.name for collection in self.client.get_collections().collections
        ]

        if collection not in existing_collections:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    @retry(**_RETRY)
    def add(self, ids, embeddings, documents=None, metadatas=None) -> None:
        """
        Добавление записей в коллекцию.
        Все передаваемые списки должны быть одинаковой длинны, иначе обрежет на самом коротком

        args:
            ids: список идентификаторов файлов ??? пока хз, надо обсудить формат передаваемых данных
            emdeddings: список эмбеддингов, соответсвующих документам
            documents: список самих документов
            metadatas: список словарей с метаданными для документов(теги и тп)

        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=to_uuid(id),
                vector=vec,
                payload={"document": doc or "", **(meta or {})},
            )
            for id, vec, doc, meta in zip(
                ids,
                embeddings,
                documents or [""] * len(ids),
                metadatas or [{}] * len(ids),
            )
        ]

        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(collection_name=self.collection, points=points[i : i + BATCH_SIZE])

    @retry(**_RETRY)
    def search(self, query_embedding, n_results=3) -> SearchResult:
        """
        Функция поиска по коллекции

        args:
            qury_embedding: эмбеддинг запроса, по которому делаем поиск
            n_results: количество ближайших/релевантных записей в возвращенном результате
            include: какие поля будет включать возвращенные записи
        """
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=n_results,
            with_payload=True,
        )
        hits = result.points

        # Qdrant возвращает cosine similarity (выше = лучше).
        # _distance_to_score ожидает дистанцию (ниже = лучше): dist = 1 - sim
        return SearchResult(
            ids=[str(h.id) for h in hits],
            documents=[(h.payload or {}).get("document", "") for h in hits],
            distances=[1.0 - h.score for h in hits],
            metadatas=[
                {k: v for k, v in (h.payload or {}).items() if k != "document"} for h in hits
            ],
        )

    @retry(**_RETRY)
    def delete(self, ids) -> None:
        """
        удаление записи/записей по ее/их id

        args:
            ids: cписок id на удаление
        """
        from qdrant_client.http.models import PointIdsList

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[to_uuid(id) for id in ids]),
        )

    @retry(**_RETRY)
    def count(self) -> int:
        """
        Возвращает количество записей в коллекции
        """
        return self.client.count(collection_name=self.collection).count

    @retry(**_RETRY)
    def get_stats(self) -> CollectionStats:
        info = self.client.get_collection(self.collection)
        return CollectionStats(
            collection=self.collection,
            vectors_count=info.points_count or 0,
            segments_count=info.segments_count,
            disk_size_bytes=getattr(info, "disk_data_size", None),
            ram_size_bytes=getattr(info, "ram_data_size", None),
        )


# ----------------- QdrantVectorStore (dense + BM25 sparse) -----------------


class QdrantVectorStore(VectorDB):
    """Qdrant collection with named dense + BM25 sparse vectors.

    Creates two named vector spaces:
        ``"dense"``  — cosine-similarity dense vectors
        ``"sparse"`` — BM25 sparse vectors (Qdrant inverted index)

    The BM25 vectorizer is fitted automatically on the first ``add()`` call
    when it is not already fitted. For best IDF accuracy, pre-fit the
    vectorizer on the full corpus and pass it via the *vectorizer* argument
    before calling ``add()``.

    Args:
        collection:  Qdrant collection name.
        vector_size: Dimensionality of the dense embeddings.
        vectorizer:  Pre-fitted or blank ``BM25SparseVectorizer``.
                     A blank instance is created when omitted.
        host:        Qdrant server host.
        port:        Qdrant HTTP port.
        in_memory:   Run Qdrant in-process (tests only).
    """

    DENSE = "dense"
    SPARSE = "sparse"

    def __init__(
        self,
        collection: str,
        vector_size: int,
        vectorizer: BM25SparseVectorizer | None = None,
        host: str = "localhost",
        port: int = 6333,
        in_memory: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import (
            Distance,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )

        self.client = QdrantClient(":memory:") if in_memory else QdrantClient(host=host, port=port)
        self.collection = collection
        self.vectorizer = vectorizer or BM25SparseVectorizer()

        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config={
                    self.DENSE: VectorParams(size=vector_size, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    self.SPARSE: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )

    @retry(**_RETRY)
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        from qdrant_client.http.models import SparseVector as QSparse
        from qdrant_client.models import PointStruct

        docs = documents or [""] * len(ids)
        metas = metadatas or [{}] * len(ids)

        # Auto-fit on first batch if the vectorizer was not pre-fitted
        if not self.vectorizer.fitted:
            self.vectorizer.fit(docs)

        points: list[PointStruct] = []
        for doc_id, vec, doc, meta in zip(ids, embeddings, docs, metas):
            sv = self.vectorizer.transform(doc)
            points.append(
                PointStruct(
                    id=to_uuid(doc_id),
                    vector={
                        self.DENSE: vec,
                        self.SPARSE: QSparse(indices=sv.indices, values=sv.values),
                    },
                    payload={"document": doc, **meta},
                )
            )

        BATCH = 100
        for i in range(0, len(points), BATCH):
            self.client.upsert(
                collection_name=self.collection, points=points[i : i + BATCH]
            )

    @retry(**_RETRY)
    def search(self, query_embedding: list[float], n_results: int = 3) -> SearchResult:
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            using=self.DENSE,
            limit=n_results,
            with_payload=True,
        )
        return self._hits_to_result(result.points)

    @retry(**_RETRY)
    def sparse_search(self, query_text: str, n_results: int = 3) -> SearchResult:
        """BM25 keyword search using the sparse vector index."""
        from qdrant_client.http.models import SparseVector as QSparse

        sv = self.vectorizer.transform(query_text)
        if not sv.indices:
            return SearchResult(ids=[], documents=[], distances=[], metadatas=[])

        result = self.client.query_points(
            collection_name=self.collection,
            query=QSparse(indices=sv.indices, values=sv.values),
            using=self.SPARSE,
            limit=n_results,
            with_payload=True,
        )
        return self._hits_to_result(result.points)

    @retry(**_RETRY)
    def delete(self, ids: list[str]) -> None:
        from qdrant_client.http.models import PointIdsList

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[to_uuid(i) for i in ids]),
        )

    @retry(**_RETRY)
    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count

    @retry(**_RETRY)
    def get_stats(self) -> CollectionStats:
        info = self.client.get_collection(self.collection)
        return CollectionStats(
            collection=self.collection,
            vectors_count=info.points_count or 0,
            segments_count=info.segments_count,
            disk_size_bytes=getattr(info, "disk_data_size", None),
            ram_size_bytes=getattr(info, "ram_data_size", None),
        )

    def _hits_to_result(self, hits: list) -> SearchResult:
        # Qdrant возвращает cosine similarity (выше = лучше).
        # _distance_to_score ожидает дистанцию (ниже = лучше): dist = 1 - sim
        return SearchResult(
            ids=[str(h.id) for h in hits],
            documents=[(h.payload or {}).get("document", "") for h in hits],
            distances=[1.0 - h.score for h in hits],
            metadatas=[
                {k: v for k, v in (h.payload or {}).items() if k != "document"}
                for h in hits
            ],
        )


# ----------------- HybridVectorStore (RRF dense + sparse) ------------------


def _rrf(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion of two ranked ID lists (1-based ranks)."""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(sparse_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class HybridVectorStore(QdrantVectorStore):
    """QdrantVectorStore extended with RRF-fused hybrid search.

    ``search()`` remains dense-only (satisfies the ``VectorDB`` contract).
    Use ``hybrid_search()`` to combine dense and BM25 results via RRF.
    """

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        n_results: int = 3,
        rrf_k: int = 60,
        fetch_k: int | None = None,
    ) -> SearchResult:
        """Dense + sparse search fused with Reciprocal Rank Fusion.

        Args:
            query_text:     Raw query string (used for BM25 sparse search).
            query_embedding: Dense embedding of the query.
            n_results:      Number of results to return after fusion.
            rrf_k:          RRF constant; higher values dampen rank differences.
            fetch_k:        Candidates per signal before fusion.
                            Defaults to ``n_results * 2``.

        Returns:
            ``SearchResult`` ranked by RRF score (descending).
        """
        fetch = fetch_k if fetch_k is not None else n_results * 2

        dense_res = self.search(query_embedding, n_results=fetch)
        sparse_res = self.sparse_search(query_text, n_results=fetch)

        fused_ids = _rrf(dense_res.ids, sparse_res.ids, k=rrf_k)[:n_results]

        # Build a lookup from both result sets to avoid extra DB round-trips
        lookup: dict[str, tuple[str, dict]] = {}
        for res in (dense_res, sparse_res):
            for doc_id, doc, meta in zip(res.ids, res.documents, res.metadatas):
                lookup.setdefault(doc_id, (doc, meta))

        ids, docs, dists, metas = [], [], [], []
        for rank, doc_id in enumerate(fused_ids, start=1):
            if doc_id not in lookup:
                continue
            doc, meta = lookup[doc_id]
            ids.append(doc_id)
            docs.append(doc)
            dists.append(1.0 / (rrf_k + rank))  # RRF score as proxy distance
            metas.append(meta)

        return SearchResult(ids=ids, documents=docs, distances=dists, metadatas=metas)
