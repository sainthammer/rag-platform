from vector_store.store_dataclasses import SearchResult
from vector_store.utils import to_uuid

from .ports import VectorDB

# ----------------- ChromaDB -----------------


class ChromaDB(VectorDB):
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
        self.collection.upsert(
            ids=[to_uuid(id) for id in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_embedding, n_results=3) -> SearchResult:
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

        self.collection.delete(ids=[to_uuid(id) for id in ids])

    def count(self) -> int:
        return self.collection.count()


# ----------------- Qdrant -----------------


class QdrantDB(VectorDB):
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
        from qdrant_client.http.models import PointIdsList

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[to_uuid(id) for id in ids]),
        )

    def count(self):
        return self.client.count(collection_name=self.collection).count
