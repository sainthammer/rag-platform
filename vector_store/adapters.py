"""
Реализации интерфейсов для базового взаимодействия с векторными БД.
"""

from vector_store.ports import VectorDB
from vector_store.store_dataclasses import SearchResult
from vector_store.utils import reciprocal_rank_fusion, to_uuid

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
            metadatas=[dict(m) for m in (raw["metadatas"] or [[]])[0]],
        )

    def delete(self, ids) -> None:
        """
        удаление записи/записей по ее/их id

        args:
            ids: cписок id на удаление
        """
        self.collection.delete(ids=[to_uuid(id) for id in ids])

    def count(self) -> int:
        """
        Возвращает количество записей в коллекции
        """
        return self.collection.count()

    def clear(self) -> None:
        """
        Очистить коллеуцеию
        """

        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)


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
        use_sparce: флаг, который бавляет sparce индекс
    """

    def __init__(
        self,
        collection: str,
        vector_size: int,
        host: str = "localhost",
        port: int = 6333,
        in_memory: bool = False,
        use_sparse: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)

        self.collection = collection
        self.use_sparse = use_sparse
        self.sparse_model = None

        if use_sparse:
            from fastembed import SparseTextEmbedding

            self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        existing_collections = [
            collection.name for collection in self.client.get_collections().collections
        ]

        if collection not in existing_collections:
            self.client.create_collection(
                collection_name=collection,
                vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()} if use_sparse else None,
            )

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
        from qdrant_client.models import PointStruct, SparseVector

        documents = documents or [""] * len(ids)
        metadatas = metadatas or [{}] * len(ids)

        sparse_vecs: SparseVector | list = [None] * len(ids)

        points = []

        for id, vec, doc, meta, sp in zip(ids, embeddings, documents, metadatas, sparse_vecs):
            vector = {"dence": vec}
            if self.use_sparse:
                vector["sparce"] = SparseVector(
                    indices=sp.indices.tolist(),
                    values=sp.values.tolist(),
                )

            points.append(
                PointStruct(
                    id=to_uuid(id), vector=vector, payload={"document": doc or "", **(meta or {})}
                )
            )

        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(collection_name=self.collection, points=points[i : i + BATCH_SIZE])

    def _to_search_result(self, hits) -> SearchResult:
        """Общий сборщик SearchResult из точек Qdrant"""
        return SearchResult(
            ids=[str(h.id) for h in hits],
            documents=[(h.payload or {}).get("document", "") for h in hits],
            distances=[h.score for h in hits],
            metadatas=[
                {k: v for k, v in (h.payload or {}).items() if k != "document"} for h in hits
            ],
        )

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
            using="dense",
            limit=n_results,
            with_payload=True,
        )
        hits = result.points

        return self._to_search_result(hits=hits)

    def sparse_search(self, query_text, n_results=3) -> SearchResult:
        """sparse search foo"""
        from qdrant_client.models import SparseVector

        if self.sparse_model is None:
            raise ValueError("sparse_search недоступен: создайте QdrantDB с use_sparse=True")

        emb = list(self.sparse_model.query_embed(query_text))[0]
        result = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            ),
            using="sparse",
            limit=n_results,
            with_payload=True,
        )

        return self._to_search_result(result.points)

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

    def count(self):
        """
        Возвращает количество записей в коллекции
        """
        return self.client.count(collection_name=self.collection).count

    def clear(self) -> None:
        """
        Удалить все точки, при этом оставив коллекцию
        """
        from qdrant_client.http.models import Filter, FilterSelector

        self.client.delete(
            collection_name=self.collection, points_selector=FilterSelector(filter=Filter())
        )


class HybridVectorStore(VectorDB):
    """
    Гибридный поиск

    Оборачивает QdrantDB с включённым sparse-индексом.
    add/delete/count делегируются обёрнутому стору.

    args:
        store: QdrantDB, созданный с use_sparse=True
        k: константа RRF
    """

    def __init__(self, store: QdrantDB, k: int = 60) -> None:
        if not store.use_sparse:
            raise ValueError("Store must be created with use_sparce=True")

        self.store = store
        self.k = k

    def add(self, ids, embeddings, documents=None, metadatas=None) -> None:
        self.store.add(ids, embeddings, documents, metadatas)

    def search(self, query_embedding, n_results=3, query_text=None) -> SearchResult:
        """
        Функция гибридного поиска. Нужен эмбединг и текст для посика
        В distances кладётся RRF-score (больше = лучше)
        """

        if query_text is None:
            raise ValueError("Hybrid search needed query_text for search")

        fetch_k = n_results * 4

        dense = self.store.search(query_embedding=query_embedding, n_results=n_results)
        sparse = self.store.sparse_search(query_text=query_text, n_results=fetch_k)

        fused = reciprocal_rank_fusion([dense.ids, sparse.ids], k=self.k)

        lookup = {}

        for res in (dense, sparse):
            for i, doc_id in enumerate(res.ids):
                lookup.setdefault(doc_id, (res.documents[i], res.metadatas[i]))

        top = fused[:n_results]

        return SearchResult(
            ids=[doc_id for doc_id, _ in top],
            documents=[lookup[doc_id][0] for doc_id, _ in top],
            distances=[score for _, score in top],
            metadatas=[lookup[doc_id][1] for doc_id, _ in top],
        )

        def delete(self, ids) -> None:
            self.store.delete(ids)

        def count(self) -> int:
            return self.store.count()
