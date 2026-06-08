"""
Пример базового взаимодействия с коллекциями + можно использовать как тест(просто запустить файл)
"""

from vector_store.adapters import ChromaDB, HybridVectorStore, QdrantDB
from vector_store.ports import VectorDB
from vector_store.utils import reciprocal_rank_fusion

EMBEDDINGS = [
    [0.1, 0.2, 0.3, 0.4],
    [0.9, 0.8, 0.7, 0.6],
    [0.5, 0.5, 0.5, 0.5],
]
IDS = ["cat", "dog", "bird"]
DOCS = ["Кошка мяукает", "Собака лает", "Птица поёт"]
METAS = [{"animal": "cat"}, {"animal": "dog"}, {"animal": "bird"}]
QUERY_VEC = [0.15, 0.25, 0.35, 0.45]


def run(db: VectorDB, label: str):
    print(f"\n{'─' * 40}")
    print(f"  {label}")
    print(f"{'─' * 40}")

    db.add(IDS, EMBEDDINGS, DOCS, METAS)
    print(f"add() count = {db.count()}")

    result = db.search(QUERY_VEC, n_results=2)
    print("search():")
    for i, (id_, doc, dist) in enumerate(zip(result.ids, result.documents, result.distances)):
        print(f"  [{i + 1}] id={id_!r:8}  dist={dist:.4f}  doc={doc!r}")

    db.delete(["cat", "bird"])
    print(f"delete('cat', 'bird') → count = {db.count()}")


def run_rrf():
    print(f"\n{'─' * 40}")
    print(f"RRF")
    print(f"{'─' * 40}")

    fused = reciprocal_rank_fusion([["b", "a", "c"], ["b", "a", "d"]])
    print("fused:", fused)
    assert fused[0][0] == "b"
    print("top-1 верный: 'b' побеждает в обоих списках")

    fused = reciprocal_rank_fusion([["x"], ["y", "z"]])
    assert set(doc_id for doc_id, _ in fused) == {"x", "y", "z"}
    print("все id попали в результат")

    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    print("пустой вход не падает")

    fused = reciprocal_rank_fusion([["only"]], k=60)
    assert abs(fused[0][1] - 1 / 61) < 1e-9
    print("формула score верна")


def run_hybrid():
    print(f"\n{'─' * 40}")
    print(f"  HybridVectorStore")
    print(f"{'─' * 40}")

    ids = ["doc-1", "doc-2", "doc-3"]
    documents = [
        "кошки любят спать на солнце",
        "питон это язык программирования",
        "qdrant хранит векторы",
    ]

    embeddings = [
        [1.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1.0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1.0, 0, 0, 0, 0, 0],
    ]
    metadatas = [{"topic": "cats"}, {"topic": "python"}, {"topic": "db"}]
    query_embedding = [0, 1.0, 0, 0, 0, 0, 0, 0]

    store = QdrantDB(collection="hybrid_smoke", vector_size=8, in_memory=True, use_sparse=True)
    hybrid = HybridVectorStore(store)

    hybrid.add(ids, embeddings, documents, metadatas)
    print(f"add() count = {hybrid.count()}")

    result = hybrid.search(query_embedding=query_embedding, query_text="питон язык", n_results=3)
    print("search():")
    for i, (id_, doc, dist) in enumerate(zip(result.ids, result.documents, result.distances)):
        print(f"  [{i + 1}] id={id_!r:8}  dist={dist:.4f}  doc={doc!r}")

    assert result.documents[0] == "питон это язык программирования"
    assert result.distances == sorted(result.distances, reverse=True)
    print("top-1 верный: 'питон' побеждает по dense + sparse")

    try:
        hybrid.search(query_embedding=query_embedding, query_text=None)
        assert False, "должно было упасть"
    except ValueError:
        print("без query_text → ValueError (ожидаемо)")


if __name__ == "__main__":
    run(ChromaDB("test_col", persist_directory="./chroma_test"), "ChromaDB")
    run(QdrantDB("test_collection", vector_size=4, in_memory=True), "QdrantDB")
    run_rrf()
    run_hybrid()
    print("\nDONE!\n")
