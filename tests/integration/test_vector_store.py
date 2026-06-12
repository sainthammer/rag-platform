"""
Интеграционные тесты для реализаций ChromaDB и QdrantDB через testcontainers.

Запуск:
    pip install pytest "testcontainers[chroma,qdrant]" chromadb qdrant-client httpx
    pip install fastembed
    pytest -v tests/integration/test_vector_store.py

Требования:
    - запущенный Docker
    - поправьте импорт ChromaDB / QdrantDB / to_uuid под структуру вашего проекта
"""

import math
import os
import time
import uuid

import chromadb
import httpx
import pytest
from testcontainers.chroma import ChromaContainer
from testcontainers.qdrant import QdrantContainer

from vector_store.adapters import ChromaDB, QdrantDB, to_uuid

for _var in ("NO_PROXY", "no_proxy"):
    _existing = os.environ.get(_var, "")
    _parts = [p for p in _existing.split(",") if p]
    for _host in ("localhost", "127.0.0.1", "::1"):
        if _host not in _parts:
            _parts.append(_host)
    os.environ[_var] = ",".join(_parts)


DIM = 8
HOST = "127.0.0.1"
CHROMA_IMAGE = f"chromadb/chroma:{chromadb.__version__}"
QDRANT_IMAGE = "qdrant/qdrant:latest"


# ----------------- readiness -----------------


def wait_http_ready(url: str, timeout: float = 90.0) -> None:
    """Поллит url, пока не получит 200. Надёжнее, чем ожидание по логам."""
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"Сервис {url} не стал готов за {timeout}s: {last_err!r}")


# ----------------- вспомогательные данные -----------------


def vec(*coords: float) -> list[float]:
    """Дополняет вектор нулями до DIM и нормализует.

    Нормализация важна для parity-теста: Chroma по умолчанию использует L2,
    Qdrant в реализации — cosine. На единичных векторах порядок ближайших
    соседей у этих метрик совпадает.
    """
    v = list(coords) + [0.0] * (DIM - len(coords))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


IDS = ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]

EMBEDDINGS = [
    vec(1.0, 0.05),  # doc-1: почти коллинеарен запросу
    vec(0.9, 0.35),  # doc-2: второй по близости
    vec(0.0, 1.0),  # doc-3: ортогонален
    vec(-1.0, 0.2),  # doc-4: противоположный
    vec(0.1, -1.0),  # doc-5
]

DOCUMENTS = [
    "Введение в машинное обучение",
    "Глубокое обучение и нейронные сети",
    "Рецепт борща со сметаной",
    "История древнего Рима",
    "Как ухаживать за кактусами",
]

METADATAS = [
    {"topic": "ai", "lang": "ru"},
    {"topic": "ai", "lang": "ru"},
    {"topic": "food", "lang": "ru"},
    {"topic": "history", "lang": "ru"},
    {"topic": "plants", "lang": "ru"},
]

QUERY = vec(1.0, 0.0)


def expected_uuid(raw_id: str) -> str:
    return str(to_uuid(raw_id))


# ----------------- контейнеры (session scope) -----------------


@pytest.fixture(scope="session")
def chroma_server():
    with ChromaContainer(CHROMA_IMAGE) as container:
        port = int(container.get_exposed_port(8000))
        # /api/v2/heartbeat для chromadb >= 1.0; для старых серверов — v1
        try:
            wait_http_ready(f"http://{HOST}:{port}/api/v2/heartbeat", timeout=60)
        except RuntimeError:
            wait_http_ready(f"http://{HOST}:{port}/api/v1/heartbeat", timeout=30)
        yield container


@pytest.fixture(scope="session")
def qdrant_server():
    with QdrantContainer(QDRANT_IMAGE) as container:
        port = int(container.get_exposed_port(6333))
        wait_http_ready(f"http://{HOST}:{port}/readyz", timeout=90)
        yield container


# ----------------- фабрики БД (свежая коллекция на каждый тест) -----------------


def make_chroma(chroma_server) -> ChromaDB:
    return ChromaDB(
        collection=f"test_{uuid.uuid4().hex[:10]}",
        host=HOST,
        port=int(chroma_server.get_exposed_port(8000)),
    )


def make_qdrant(qdrant_server, use_sparse: bool = False) -> QdrantDB:
    return QdrantDB(
        collection=f"test_{uuid.uuid4().hex[:10]}",
        vector_size=DIM,
        host=HOST,
        port=int(qdrant_server.get_exposed_port(6333)),
        use_sparse=use_sparse,
    )


@pytest.fixture(params=["chroma", "qdrant"])
def db(request, chroma_server, qdrant_server):
    """Параметризованная фикстура: каждый тест выполняется против обеих БД."""
    if request.param == "chroma":
        yield make_chroma(chroma_server)
    else:
        yield make_qdrant(qdrant_server)


@pytest.fixture
def populated_db(db):
    db.add(ids=IDS, embeddings=EMBEDDINGS, documents=DOCUMENTS, metadatas=METADATAS)
    return db


# ----------------- общие тесты (одинаковые запросы к обеим БД) -----------------


def test_add_and_count(populated_db):
    assert populated_db.count() == len(IDS)


def test_add_is_upsert(populated_db):
    """Повторное добавление тех же id не должно дублировать записи."""
    populated_db.add(
        ids=IDS[:2],
        embeddings=EMBEDDINGS[:2],
        documents=["обновлённый документ 1", "обновлённый документ 2"],
        metadatas=METADATAS[:2],
    )
    assert populated_db.count() == len(IDS)


def test_search_returns_n_results(populated_db):
    result = populated_db.search(QUERY, n_results=3)
    assert len(result.ids) == 3
    assert len(result.documents) == 3
    assert len(result.distances) == 3
    assert len(result.metadatas) == 3


def test_search_nearest_order(populated_db):
    """Ближайшие к запросу — doc-1, затем doc-2, у обеих реализаций."""
    result = populated_db.search(QUERY, n_results=2)
    assert result.ids == [expected_uuid("doc-1"), expected_uuid("doc-2")]


def test_search_returns_documents_and_metadata(populated_db):
    result = populated_db.search(QUERY, n_results=1)
    assert result.documents[0] == DOCUMENTS[0]
    # сравниваем как надмножество: реализации не должны терять ключи
    for key, value in METADATAS[0].items():
        assert result.metadatas[0].get(key) == value


def test_search_n_results_larger_than_collection(populated_db):
    result = populated_db.search(QUERY, n_results=50)
    assert len(result.ids) <= len(IDS)


def test_delete(populated_db):
    populated_db.delete(["doc-1"])
    assert populated_db.count() == len(IDS) - 1

    result = populated_db.search(QUERY, n_results=len(IDS))
    assert expected_uuid("doc-1") not in result.ids
    assert result.ids[0] == expected_uuid("doc-2")


def test_delete_multiple(populated_db):
    populated_db.delete(["doc-1", "doc-3", "doc-5"])
    assert populated_db.count() == 2


def test_clear(populated_db):
    populated_db.clear()
    assert populated_db.count() == 0

    populated_db.add(ids=["doc-new"], embeddings=[vec(0.5, 0.5)], documents=["новый"])
    assert populated_db.count() == 1


# ----------------- parity: прямое сравнение Chroma vs Qdrant -----------------


def test_parity_same_ranking(chroma_server, qdrant_server):
    """Один и тот же запрос к обеим БД даёт одинаковый порядок id."""
    chroma = make_chroma(chroma_server)
    qdrant = make_qdrant(qdrant_server)

    for impl in (chroma, qdrant):
        impl.add(ids=IDS, embeddings=EMBEDDINGS, documents=DOCUMENTS, metadatas=METADATAS)

    queries = [
        QUERY,
        vec(0.0, 1.0),
        vec(-0.7, 0.7),
        vec(0.3, -0.9, 0.1),
    ]

    for q in queries:
        chroma_result = chroma.search(q, n_results=len(IDS))
        qdrant_result = qdrant.search(q, n_results=len(IDS))

        assert chroma_result.ids == qdrant_result.ids, (
            f"Разное ранжирование для запроса {q}:\n"
            f"  chroma: {chroma_result.ids}\n"
            f"  qdrant: {qdrant_result.ids}"
        )
        assert chroma_result.documents == qdrant_result.documents


def test_parity_distances_monotonic(chroma_server, qdrant_server):
    """Семантика скоров разная (Chroma — distance, меньше=лучше; Qdrant — score,
    больше=лучше), но и там и там значения должны быть монотонны по рангу."""
    chroma = make_chroma(chroma_server)
    qdrant = make_qdrant(qdrant_server)

    for impl in (chroma, qdrant):
        impl.add(ids=IDS, embeddings=EMBEDDINGS, documents=DOCUMENTS, metadatas=METADATAS)

    chroma_result = chroma.search(QUERY, n_results=len(IDS))
    qdrant_result = qdrant.search(QUERY, n_results=len(IDS))

    assert chroma_result.distances == sorted(chroma_result.distances)
    assert qdrant_result.distances == sorted(qdrant_result.distances, reverse=True)


# ----------------- специфичные возможности -----------------


def test_qdrant_sparse_search(qdrant_server):
    pytest.importorskip("fastembed")

    qdrant = make_qdrant(qdrant_server, use_sparse=True)
    qdrant.add(ids=IDS, embeddings=EMBEDDINGS, documents=DOCUMENTS, metadatas=METADATAS)

    result = qdrant.sparse_search("машинное обучение", n_results=2)
    assert len(result.ids) == 2
    assert result.ids[0] in {expected_uuid("doc-1"), expected_uuid("doc-2")}


def test_qdrant_sparse_search_raises_without_flag(qdrant_server):
    qdrant = make_qdrant(qdrant_server, use_sparse=False)
    with pytest.raises(ValueError):
        qdrant.sparse_search("любой запрос")
