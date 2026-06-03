"""
Реализации интерфейсов для базового взаимодействия с векторными БД.
"""

from vector_store.store_dataclasses import SearchResult
from vector_store.utils import to_uuid

from .ports import VectorDB

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
            metadatas=(raw["metadatas"] or [[]])[0],
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
        from qdrant_client.http.models import PointStruct

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

        return SearchResult(
            ids=[str(h.id) for h in hits],
            documents=[h.payload.get("document", "") for h in hits],
            distances=[h.score for h in hits],
            metadatas=[{k: v for k, v in h.payload.items() if k != "document"} for h in hits],
        )

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
