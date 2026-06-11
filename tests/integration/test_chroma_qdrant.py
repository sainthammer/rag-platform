"""Интеграционные тесты: Chroma vs Qdrant — одни документы, одни запросы.

Структура:
  Блок 1 (in-memory, без Docker)
    — ChromaDB (PersistentClient через tmp_path)
    — QdrantVectorStore(in_memory=True)
    — HybridVectorStore(in_memory=True)
    Запускаются всегда: не нужны внешние сервисы.

  Блок 2 (testcontainers, требует Docker)
    — QdrantContainer("qdrant/qdrant:latest")
    Помечены @pytest.mark.integration и пропускаются,
    если Docker недоступен или testcontainers не установлен.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from embeddings.adapters import FakeEmbeddingService
from vector_store.adapters import ChromaDB, HybridVectorStore, QdrantVectorStore
from vector_store.bm25 import BM25SparseVectorizer
from vector_store.store_dataclasses import SearchResult

# ---------------------------------------------------------------------------
# testcontainers — опциональная зависимость
# ---------------------------------------------------------------------------

try:
    from testcontainers.qdrant import QdrantContainer as _QdrantContainer

    _HAS_TC = True
except ImportError:
    _HAS_TC = False


def _docker_available() -> bool:
    """Проверить, доступен ли Docker-демон без броска исключения."""
    if not _HAS_TC:
        return False
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


_DOCKER_OK = _docker_available()

# ---------------------------------------------------------------------------
# Общие константы
# ---------------------------------------------------------------------------

_EMBED = FakeEmbeddingService(size=8, normalize=True)

_DOCS: dict[str, str] = {
    "py":     "Python is a high-level programming language for data science and ML.",
    "docker": "Docker containers isolate applications and their dependencies.",
    "ml":     "Machine learning models require large datasets and significant compute.",
    "api":    "REST APIs use HTTP verbs to expose resources over the network.",
    "db":     "Databases store and retrieve structured data efficiently.",
}


def _add_docs(store: Any, docs: dict[str, str]) -> None:
    ids = list(docs.keys())
    texts = list(docs.values())
    store.add(
        ids=ids,
        embeddings=[_EMBED.embed(t) for t in texts],
        documents=texts,
        metadatas=[{"source": f"{k}.txt"} for k in ids],
    )


def _fitted_vectorizer() -> BM25SparseVectorizer:
    return BM25SparseVectorizer().fit(list(_DOCS.values()))


# ---------------------------------------------------------------------------
# Блок 1: in-memory тесты (без Docker)
# ---------------------------------------------------------------------------


def test_bm25_vectorizer_fit_transform() -> None:
    """BM25-векторизатор должен давать непустой SparseVector для слов из корпуса."""
    v = _fitted_vectorizer()
    sv = v.transform("machine learning datasets compute")
    assert sv.indices, "Ожидаем непустой sparse vector"
    assert len(sv.indices) == len(sv.values)
    assert all(w > 0 for w in sv.values), "Все веса должны быть положительными"


def test_bm25_oov_returns_empty() -> None:
    """Слова вне словаря → пустой SparseVector, без ошибок."""
    v = _fitted_vectorizer()
    sv = v.transform("xyzzy quux frobnicator")
    assert sv.indices == []
    assert sv.values == []


def test_bm25_not_fitted_returns_empty() -> None:
    """transform() до fit() должен вернуть пустой SparseVector, а не упасть."""
    v = BM25SparseVectorizer()
    sv = v.transform("some text")
    assert sv.indices == []


def test_chroma_exact_match_top1(tmp_path: Path) -> None:
    """ChromaDB: точный текст документа как запрос → top-1 совпадает, distance ≈ 0.

    ChromaDB возвращает L2-расстояние (0 = идентичные векторы),
    в отличие от Qdrant, который возвращает cosine similarity (1.0 = идентичные).
    """
    db = ChromaDB("exact_chroma", persist_directory=str(tmp_path / "chroma"))
    _add_docs(db, _DOCS)

    target = _DOCS["py"]
    res: SearchResult = db.search(_EMBED.embed(target), n_results=3)

    assert res.documents, "Chroma вернула пустой результат"
    assert res.documents[0] == target, f"top-1 != target: {res.documents[0]!r}"
    # L2-расстояние: 0.0 = точное совпадение, < 0.1 = очень близко
    assert res.distances[0] < 0.1, f"L2-distance слишком высокая: {res.distances[0]}"


def test_qdrant_inmemory_exact_match_top1() -> None:
    """QdrantVectorStore(in_memory): точный текст → top-1 с низкой дистанцией.

    Qdrant возвращает cosine similarity; адаптер конвертирует в дистанцию (1 - sim),
    поэтому точное совпадение → dist ≈ 0.0 (аналогично L2 у Chroma).
    """
    store = QdrantVectorStore(
        "exact_qdrant", vector_size=8, vectorizer=_fitted_vectorizer(), in_memory=True
    )
    _add_docs(store, _DOCS)

    target = _DOCS["ml"]
    res = store.search(_EMBED.embed(target), n_results=3)

    assert res.documents, "Qdrant вернул пустой результат"
    assert res.documents[0] == target, f"top-1 != target: {res.documents[0]!r}"
    assert res.distances[0] < 0.1, f"дистанция слишком высокая: {res.distances[0]}"


def test_chroma_and_qdrant_top1_match(tmp_path: Path) -> None:
    """Chroma и Qdrant должны возвращать одинаковый top-1 на точном запросе."""
    vectorizer = _fitted_vectorizer()

    chroma = ChromaDB("match_chroma", persist_directory=str(tmp_path / "chroma"))
    qdrant = QdrantVectorStore("match_qdrant", vector_size=8, vectorizer=vectorizer, in_memory=True)

    _add_docs(chroma, _DOCS)
    _add_docs(qdrant, _DOCS)

    for key, text in _DOCS.items():
        emb = _EMBED.embed(text)
        c_res = chroma.search(emb, n_results=1)
        q_res = qdrant.search(emb, n_results=1)

        assert c_res.documents, f"Chroma: нет результатов для '{key}'"
        assert q_res.documents, f"Qdrant: нет результатов для '{key}'"
        assert c_res.documents[0] == text, f"Chroma top-1 для '{key}' не совпал"
        assert q_res.documents[0] == text, f"Qdrant top-1 для '{key}' не совпал"


def test_result_sets_overlap_inmemory(tmp_path: Path) -> None:
    """Top-3 результатов Chroma и Qdrant должны пересекаться хотя бы по 1 документу."""
    vectorizer = _fitted_vectorizer()

    chroma = ChromaDB("overlap_chroma", persist_directory=str(tmp_path / "chroma"))
    qdrant = QdrantVectorStore("overlap_qdrant", vector_size=8, vectorizer=vectorizer, in_memory=True)

    _add_docs(chroma, _DOCS)
    _add_docs(qdrant, _DOCS)

    emb = _EMBED.embed(_DOCS["api"])
    c_docs = set(chroma.search(emb, n_results=3).documents)
    q_docs = set(qdrant.search(emb, n_results=3).documents)

    assert c_docs & q_docs, "Top-3 Chroma и Qdrant не пересекаются"


def test_qdrant_sparse_search_finds_keyword() -> None:
    """sparse_search() должен включить документ, содержащий ключевые слова запроса."""
    store = QdrantVectorStore(
        "sparse_inmem", vector_size=8, vectorizer=_fitted_vectorizer(), in_memory=True
    )
    _add_docs(store, _DOCS)

    res = store.sparse_search("machine learning datasets", n_results=3)

    assert res.documents, "sparse_search вернул пустой результат"
    assert _DOCS["ml"] in res.documents, (
        f"ML-документ не найден в sparse top-3: {res.documents}"
    )


def test_qdrant_delete_and_count() -> None:
    """delete() уменьшает count(); повторное удаление — без ошибок."""
    store = QdrantVectorStore(
        "del_test", vector_size=8, vectorizer=_fitted_vectorizer(), in_memory=True
    )
    _add_docs(store, _DOCS)

    before = store.count()
    assert before == len(_DOCS)

    store.delete(["py", "docker"])
    assert store.count() == before - 2

    # повторное удаление не должно падать
    store.delete(["py"])
    assert store.count() == before - 2


def test_hybrid_search_includes_best_match() -> None:
    """hybrid_search() должен включить документ, совпадающий по обоим сигналам."""
    store = HybridVectorStore(
        "hybrid_inmem", vector_size=8, vectorizer=_fitted_vectorizer(), in_memory=True
    )
    _add_docs(store, _DOCS)

    target = _DOCS["db"]
    res = store.hybrid_search(
        query_text="databases structured data",
        query_embedding=_EMBED.embed(target),
        n_results=3,
    )

    assert res.documents, "hybrid_search вернул пустой результат"
    assert target in res.documents, (
        f"Целевой документ не найден в гибридном top-3: {res.documents}"
    )


def test_hybrid_search_rrf_scores_descending() -> None:
    """RRF-score в hybrid_search должны убывать (или быть равными) по индексу."""
    store = HybridVectorStore(
        "hybrid_rrf_order", vector_size=8, vectorizer=_fitted_vectorizer(), in_memory=True
    )
    _add_docs(store, _DOCS)

    res = store.hybrid_search(
        query_text="python programming",
        query_embedding=_EMBED.embed(_DOCS["py"]),
        n_results=5,
    )

    assert res.distances == sorted(res.distances, reverse=True), (
        "RRF-score должны убывать: " + str(res.distances)
    )


def test_hybrid_store_dense_fallback_compatible_with_vdb() -> None:
    """HybridVectorStore.search() работает как чисто плотный поиск (совместимость с VectorDB)."""
    store = HybridVectorStore(
        "hybrid_dense_fb", vector_size=8, vectorizer=_fitted_vectorizer(), in_memory=True
    )
    _add_docs(store, _DOCS)

    target = _DOCS["api"]
    res = store.search(_EMBED.embed(target), n_results=1)

    assert res.documents
    assert res.documents[0] == target


def test_chroma_metadata_filter(tmp_path: Path) -> None:
    """ChromaDB filters по metadata возвращают только подходящие документы."""
    from retrieval.retriever import Retriever

    db = ChromaDB("filter_meta", persist_directory=str(tmp_path / "chroma"))
    _add_docs(db, _DOCS)

    retriever = Retriever(embed_fn=_EMBED.embed, vector_db_factory=lambda _: db)
    results = retriever.retrieve(
        "programming", collection="filter_meta", top_k=5, filters={"source": "py.txt"}
    )

    assert results, "Фильтр вернул пустой список"
    assert all(r.metadata.get("source") == "py.txt" for r in results)


def test_qdrant_upsert_is_idempotent() -> None:
    """Повторный add() с теми же id не дублирует записи (upsert-семантика)."""
    store = QdrantVectorStore(
        "upsert_test", vector_size=8, vectorizer=_fitted_vectorizer(), in_memory=True
    )
    _add_docs(store, _DOCS)
    count_first = store.count()

    # Добавляем те же документы повторно
    _add_docs(store, _DOCS)
    assert store.count() == count_first, "Upsert не должен увеличивать count"


# ---------------------------------------------------------------------------
# Блок 2: testcontainers (требует Docker)
# ---------------------------------------------------------------------------

_tc_skip = pytest.mark.skipif(
    not _DOCKER_OK,
    reason="Docker недоступен — пропускаем testcontainers-тесты",
)


@pytest.fixture(scope="module")
def qdrant_container():
    with _QdrantContainer("qdrant/qdrant:latest") as c:
        yield c


def _conn(container) -> dict:
    return {
        "host": container.get_container_host_ip(),
        "port": int(container.get_exposed_port(6333)),
    }


@_tc_skip
@pytest.mark.integration
def test_tc_qdrant_exact_match(tmp_path: Path, qdrant_container) -> None:
    """[Docker] QdrantVectorStore через реальный контейнер: exact-match top-1.

    Адаптер хранит dist = 1 - cosine_sim, поэтому точное совпадение → dist ≈ 0.
    """
    store = QdrantVectorStore(
        "tc_exact", vector_size=8, vectorizer=_fitted_vectorizer(), **_conn(qdrant_container)
    )
    _add_docs(store, _DOCS)

    target = _DOCS["py"]
    res = store.search(_EMBED.embed(target), n_results=3)

    assert res.documents[0] == target
    assert res.distances[0] < 0.1


@_tc_skip
@pytest.mark.integration
def test_tc_hybrid_search(tmp_path: Path, qdrant_container) -> None:
    """[Docker] HybridVectorStore через реальный контейнер: RRF fusion."""
    store = HybridVectorStore(
        "tc_hybrid", vector_size=8, vectorizer=_fitted_vectorizer(), **_conn(qdrant_container)
    )
    _add_docs(store, _DOCS)

    res = store.hybrid_search(
        query_text="databases structured data",
        query_embedding=_EMBED.embed(_DOCS["db"]),
        n_results=3,
    )

    assert _DOCS["db"] in res.documents


@_tc_skip
@pytest.mark.integration
def test_tc_chroma_vs_qdrant_overlap(tmp_path: Path, qdrant_container) -> None:
    """[Docker] Chroma и Qdrant-контейнер возвращают пересекающийся top-3."""
    chroma = ChromaDB("tc_overlap", persist_directory=str(tmp_path / "chroma"))
    qdrant = QdrantVectorStore(
        "tc_overlap_q", vector_size=8, vectorizer=_fitted_vectorizer(), **_conn(qdrant_container)
    )

    _add_docs(chroma, _DOCS)
    _add_docs(qdrant, _DOCS)

    emb = _EMBED.embed(_DOCS["ml"])
    c_docs = set(chroma.search(emb, n_results=3).documents)
    q_docs = set(qdrant.search(emb, n_results=3).documents)

    assert c_docs & q_docs, "Результаты Chroma и контейнерного Qdrant не пересекаются"
