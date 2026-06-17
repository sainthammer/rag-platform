"""
Юнит-тесты для ChromaDB, QdrantDB и HybridVectorStore.
"""

from unittest.mock import MagicMock

import pytest

from vector_store.adapters import ChromaDB, HybridVectorStore, QdrantDB
from vector_store.store_dataclasses import (
    MetadataFilter,
    MetadataScheme,
    SearchResult,
    combine_filters,
)

# -------------- Fixtures --------------


@pytest.fixture
def chroma():
    db = ChromaDB.__new__(ChromaDB)
    db.client = MagicMock()
    db.collection = MagicMock()
    db.collection.name = "test_col"
    return db


@pytest.fixture
def qdrant():
    db = QdrantDB.__new__(QdrantDB)
    db.client = MagicMock()
    db.collection = "test_col"
    db.use_sparse = False
    db.sparse_model = None
    return db


@pytest.fixture
def qdrant_sparse():
    db = QdrantDB.__new__(QdrantDB)
    db.client = MagicMock()
    db.collection = "test_col"
    db.use_sparse = True
    db.sparse_model = MagicMock()
    return db


@pytest.fixture
def hybrid(qdrant_sparse):
    return HybridVectorStore(qdrant_sparse, k=60), qdrant_sparse


def make_hits(*args):
    hits = []
    for id_, score, doc, meta in args:
        h = MagicMock()
        h.id = id_
        h.score = score
        h.payload = {"document": doc, **meta}
        hits.append(h)
    return hits


def make_sr(ids, docs=None, dists=None, metas=None):
    n = len(ids)
    return SearchResult(ids, docs or ["d"] * n, dists or [0.9] * n, metas or [{}] * n)


# -------------- ChromaDB --------------


class TestChromaDBAdd:
    def test_calls_upsert_with_correct_args(self, chroma):
        from vector_store.utils import to_uuid

        chroma.add(["a", "b"], [[0.1], [0.2]], ["d1", "d2"], [{"x": 1}, {}])
        chroma.collection.upsert.assert_called_once_with(
            ids=[to_uuid("a"), to_uuid("b")],
            embeddings=[[0.1], [0.2]],
            documents=["d1", "d2"],
            metadatas=[{"x": 1}, {}],
        )


class TestChromaDBSearch:
    @pytest.fixture(autouse=True)
    def _setup(self, chroma):
        self.db = chroma
        chroma.collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["hello", "world"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"k": "v"}, {}]],
        }

    def test_returns_search_result(self):
        r = self.db.search([0.5], n_results=2)
        assert r.ids == ["id1", "id2"]
        assert r.documents == ["hello", "world"]
        assert r.distances == pytest.approx([0.1, 0.2])
        assert r.metadatas[0] == {"k": "v"}

    def test_passes_filters_to_where(self):
        f = MetadataFilter(field="source", op="eq", value="web")
        self.db.search([0.0], filters=[f])
        kwargs = self.db.collection.query.call_args.kwargs
        assert kwargs["where"] == {"source": {"$eq": "web"}}

    def test_handles_none_fields(self, chroma):
        chroma.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": None,
            "distances": None,
            "metadatas": None,
        }
        r = chroma.search([0.0])
        assert r.documents == []
        assert r.distances == []


class TestChromaDBClear:
    def test_deletes_and_recreates_collection(self, chroma):
        new_col = MagicMock()
        chroma.client.get_or_create_collection.return_value = new_col
        chroma.clear()
        chroma.client.delete_collection.assert_called_once_with("test_col")
        assert chroma.collection is new_col


# -------------- QdrantDB --------------


class TestQdrantDBSearch:
    def test_returns_correct_fields(self, qdrant):
        qdrant.client.query_points.return_value.points = make_hits(
            ("1", 0.9, "hello", {"tag": "a"}), ("2", 0.7, "world", {})
        )
        r = qdrant.search([0.1], n_results=2)
        assert r.ids == ["1", "2"]
        assert r.documents == ["hello", "world"]
        assert r.distances == pytest.approx([0.9, 0.7])

    def test_strips_document_from_metadata(self, qdrant):
        qdrant.client.query_points.return_value.points = make_hits(
            ("1", 0.5, "text", {"author": "Alice"})
        )
        r = qdrant.search([0.0])
        assert "document" not in r.metadatas[0]
        assert r.metadatas[0]["author"] == "Alice"


class TestQdrantDBSparseSearch:
    def test_raises_without_sparse_model(self, qdrant):
        with pytest.raises(ValueError, match="use_sparse=True"):
            qdrant.sparse_search("query")

    def test_uses_sparse_index(self, qdrant_sparse):
        fake_emb = MagicMock()
        fake_emb.indices.tolist.return_value = [0, 1]
        fake_emb.values.tolist.return_value = [0.5, 0.5]
        qdrant_sparse.sparse_model.query_embed.return_value = [fake_emb]
        qdrant_sparse.client.query_points.return_value.points = []

        qdrant_sparse.sparse_search("test query", n_results=3)

        _, kwargs = qdrant_sparse.client.query_points.call_args
        assert kwargs["using"] == "sparse"
        assert kwargs["limit"] == 3


class TestQdrantDBAdd:
    def test_batches_by_100(self, qdrant):
        qdrant.add(ids=[str(i) for i in range(150)], embeddings=[[float(i)] for i in range(150)])
        assert qdrant.client.upsert.call_count == 2


# -------------- HybridVectorStore --------------


class TestHybridVectorStoreInit:
    def test_raises_when_sparse_disabled(self, qdrant):
        with pytest.raises(ValueError):
            HybridVectorStore(qdrant)


class TestHybridVectorStoreSearch:
    def test_raises_without_query_text(self, hybrid):
        h, _ = hybrid
        with pytest.raises(ValueError, match="query_text"):
            h.search([0.1])

    def test_rrf_promotes_shared_id(self, hybrid):
        h, store = hybrid
        store.search = MagicMock(return_value=make_sr(["a", "b"]))
        store.sparse_search = MagicMock(return_value=make_sr(["b", "c"]))
        r = h.search([0.0], n_results=3, query_text="q")
        assert r.ids[0] == "b"

    def test_fetch_k_is_n_results_times_4(self, hybrid):
        h, store = hybrid
        store.search = MagicMock(return_value=make_sr(["a"]))
        store.sparse_search = MagicMock(return_value=make_sr(["a"]))
        h.search([0.0], n_results=3, query_text="q")
        _, kwargs = store.sparse_search.call_args
        assert kwargs["n_results"] == 12


class TestHybridVectorStoreDelegation:
    def test_add_delete_count_delegate_to_store(self, hybrid):
        h, store = hybrid
        store.add = MagicMock()
        store.delete = MagicMock()
        store.count = MagicMock(return_value=7)

        h.add(["1"], [[0.1]])
        h.delete(["1"])

        store.add.assert_called_once()
        store.delete.assert_called_once_with(["1"])
        assert h.count() == 7


# -------------- MetadataScheme --------------


class TestMetadataScheme:
    @pytest.fixture
    def scheme(self):
        return MetadataScheme(source_file="doc.pdf", page_num=2, chunk_id=5, language="ru")

    def test_to_dict_contains_all_fields(self, scheme):
        d = scheme.to_dict()
        assert d["source_file"] == "doc.pdf"
        assert d["page_num"] == 2
        assert d["chunk_id"] == 5
        assert d["language"] == "ru"
        assert d["section"] == ""

    def test_created_at_is_iso_string(self, scheme):
        d = scheme.to_dict()
        # должна парситься обратно без ошибок
        from datetime import datetime

        datetime.fromisoformat(d["created_at"])


# -------------- MetadataFilter --------------


class TestMetadataFilter:
    def test_to_chroma_where_maps_operator(self):
        f = MetadataFilter(field="language", op="eq", value="ru")
        assert f.to_chroma_where() == {"language": {"$eq": "ru"}}

    def test_to_chroma_where_in_operator(self):
        f = MetadataFilter(field="language", op="in", value=["ru", "en"])
        assert f.to_chroma_where() == {"language": {"$in": ["ru", "en"]}}

    def test_unknown_operator_raises(self):
        f = MetadataFilter(field="language", op="like", value="ru")
        with pytest.raises(ValueError, match="Unknown operator"):
            f.to_chroma_where()


# -------------- combine_filters --------------


class TestCombineFilters:
    def test_empty_list_returns_none(self):
        assert combine_filters([]) is None

    def test_single_filter_returns_flat_dict(self):
        f = MetadataFilter(field="language", op="eq", value="ru")
        assert combine_filters([f]) == {"language": {"$eq": "ru"}}

    def test_multiple_filters_wrapped_in_and(self):
        filters = [
            MetadataFilter(field="language", op="eq", value="ru"),
            MetadataFilter(field="page_num", op="gte", value=3),
        ]
        result = combine_filters(filters)
        assert result == {
            "$and": [
                {"language": {"$eq": "ru"}},
                {"page_num": {"$gte": 3}},
            ]
        }
