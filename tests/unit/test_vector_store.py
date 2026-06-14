"""Unit tests for vector_store/ — ChromaDB, QdrantDB, QdrantVectorStore, HybridVectorStore.

ChromaDB uses PersistentClient(path=tmp_path) — no external services.
Qdrant adapters use in_memory=True — no external services.
"""

from __future__ import annotations

import math
import random

import pytest

from vector_store.adapters import ChromaDB, HybridVectorStore, QdrantVectorStore, _rrf
from vector_store.store_dataclasses import CollectionStats, SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_SIZE = 8


def _vecs(n: int, seed: int = 42) -> list[list[float]]:
    """Return n deterministic L2-normalized vectors of VECTOR_SIZE dimensions."""
    rng = random.Random(seed)
    result = []
    for _ in range(n):
        v = [rng.gauss(0, 1) for _ in range(VECTOR_SIZE)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        result.append([x / norm for x in v])
    return result


# ---------------------------------------------------------------------------
# _rrf (standalone function)
# ---------------------------------------------------------------------------


class TestRRF:
    def test_single_list(self):
        ids = ["a", "b", "c"]
        fused = _rrf(ids, [])
        assert fused == ids

    def test_merge_two_lists(self):
        dense = ["a", "b", "c"]
        sparse = ["c", "a", "d"]
        fused = _rrf(dense, sparse)
        # "a" appears in both → should rank higher than "b" or "d"
        assert fused[0] in {"a", "c"}
        assert set(fused) == {"a", "b", "c", "d"}

    def test_empty_both(self):
        assert _rrf([], []) == []

    def test_ordering_by_shared_rank(self):
        dense = ["x", "y"]
        sparse = ["x", "z"]
        fused = _rrf(dense, sparse)
        # "x" is rank-1 in both → highest score
        assert fused[0] == "x"


# ---------------------------------------------------------------------------
# CollectionStats dataclass
# ---------------------------------------------------------------------------


class TestCollectionStats:
    def test_required_fields(self):
        s = CollectionStats(collection="col", vectors_count=5)
        assert s.collection == "col"
        assert s.vectors_count == 5
        assert s.segments_count is None
        assert s.disk_size_bytes is None
        assert s.ram_size_bytes is None

    def test_optional_fields(self):
        s = CollectionStats(
            collection="col",
            vectors_count=10,
            segments_count=2,
            disk_size_bytes=1024,
            ram_size_bytes=512,
        )
        assert s.segments_count == 2
        assert s.disk_size_bytes == 1024
        assert s.ram_size_bytes == 512


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------


class TestChromaDB:
    @pytest.fixture
    def db(self, tmp_path):
        return ChromaDB("test_col", persist_directory=str(tmp_path))

    def test_add_and_count(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["doc a", "doc b", "doc c"])
        assert db.count() == 3

    def test_count_empty(self, db):
        assert db.count() == 0

    def test_search_returns_search_result(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["doc a", "doc b", "doc c"])
        result = db.search(vecs[0], n_results=2)
        assert isinstance(result, SearchResult)
        assert len(result.ids) == 2
        assert len(result.documents) == 2
        assert len(result.distances) == 2
        assert len(result.metadatas) == 2

    def test_search_top_result_matches_query(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["doc a", "doc b", "doc c"])
        result = db.search(vecs[1], n_results=1)
        assert result.documents[0] == "doc b"

    def test_delete_reduces_count(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["doc a", "doc b", "doc c"])
        db.delete(["a"])
        assert db.count() == 2

    def test_upsert_idempotent(self, db):
        vecs = _vecs(2)
        db.add(["a", "b"], vecs, ["doc a", "doc b"])
        db.add(["a"], [vecs[0]], ["doc a v2"])
        assert db.count() == 2

    def test_add_with_metadatas(self, db):
        vecs = _vecs(2)
        db.add(["a", "b"], vecs, ["doc a", "doc b"], [{"tag": "x"}, {"tag": "y"}])
        result = db.search(vecs[0], n_results=2)
        tags = {m.get("tag") for m in result.metadatas}
        assert tags == {"x", "y"}

    def test_get_stats(self, db):
        vecs = _vecs(2)
        db.add(["a", "b"], vecs, ["doc a", "doc b"])
        stats = db.get_stats()
        assert isinstance(stats, CollectionStats)
        assert stats.vectors_count == 2
        assert stats.collection == "test_col"
        assert stats.segments_count is None  # ChromaDB doesn't expose this

    def test_get_stats_empty(self, db):
        stats = db.get_stats()
        assert stats.vectors_count == 0


# ---------------------------------------------------------------------------
# QdrantVectorStore (dense + BM25 sparse)
# ---------------------------------------------------------------------------


class TestQdrantVectorStore:
    @pytest.fixture
    def db(self):
        return QdrantVectorStore("test_hybrid", vector_size=VECTOR_SIZE, in_memory=True)

    def test_add_and_count(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["hello world", "python code", "vector search"])
        assert db.count() == 3

    def test_search_dense(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["doc a", "doc b", "doc c"])
        result = db.search(vecs[0], n_results=2)
        assert isinstance(result, SearchResult)
        assert len(result.ids) == 2

    def test_sparse_search_returns_result(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["hello world foo", "python code bar", "vector search baz"])
        result = db.sparse_search("hello world", n_results=2)
        assert isinstance(result, SearchResult)

    def test_sparse_search_empty_query(self, db):
        vecs = _vecs(2)
        db.add(["a", "b"], vecs, ["doc a", "doc b"])
        # Query with no known terms → empty result (not an error)
        result = db.sparse_search("", n_results=2)
        assert isinstance(result, SearchResult)
        assert result.ids == []

    def test_delete(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["doc a", "doc b", "doc c"])
        db.delete(["b"])
        assert db.count() == 2

    def test_get_stats(self, db):
        vecs = _vecs(2)
        db.add(["a", "b"], vecs, ["doc a", "doc b"])
        stats = db.get_stats()
        assert isinstance(stats, CollectionStats)
        assert stats.vectors_count == 2

    def test_vectorizer_auto_fit(self, db):
        vecs = _vecs(2)
        assert not db.vectorizer.fitted
        db.add(["a", "b"], vecs, ["hello world", "foo bar"])
        assert db.vectorizer.fitted


# ---------------------------------------------------------------------------
# HybridVectorStore
# ---------------------------------------------------------------------------


class TestHybridVectorStore:
    @pytest.fixture
    def db(self):
        return HybridVectorStore("test_rrf", vector_size=VECTOR_SIZE, in_memory=True)

    def test_hybrid_search_returns_result(self, db):
        vecs = _vecs(4)
        db.add(
            ["a", "b", "c", "d"],
            vecs,
            ["python fastapi web", "vector database search", "rag pipeline llm", "machine learning"],
        )
        result = db.hybrid_search("python web", vecs[0], n_results=3)
        assert isinstance(result, SearchResult)
        assert len(result.ids) <= 3

    def test_hybrid_search_ids_are_strings(self, db):
        vecs = _vecs(2)
        db.add(["a", "b"], vecs, ["doc a text", "doc b text"])
        result = db.hybrid_search("doc", vecs[0], n_results=2)
        assert all(isinstance(i, str) for i in result.ids)

    def test_hybrid_search_distances_are_floats(self, db):
        vecs = _vecs(2)
        db.add(["a", "b"], vecs, ["hello world", "foo bar"])
        result = db.hybrid_search("hello", vecs[0], n_results=2)
        assert all(isinstance(d, float) for d in result.distances)

    def test_hybrid_search_respects_n_results(self, db):
        vecs = _vecs(5)
        db.add(["a", "b", "c", "d", "e"], vecs, [f"doc {i}" for i in range(5)])
        result = db.hybrid_search("doc", vecs[0], n_results=2)
        assert len(result.ids) <= 2

    def test_inherits_dense_search(self, db):
        vecs = _vecs(3)
        db.add(["a", "b", "c"], vecs, ["doc a", "doc b", "doc c"])
        result = db.search(vecs[0], n_results=1)
        assert isinstance(result, SearchResult)
        assert len(result.ids) == 1
