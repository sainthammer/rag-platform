"""
Пример базового взаимодействия с коллекциями + можно использовать как тест(просто запустить этот файл)
"""

from vector_store.adapters import ChromaDB, QdrantDB

EMBEDDINGS = [
    [0.1, 0.2, 0.3, 0.4],
    [0.9, 0.8, 0.7, 0.6],
    [0.5, 0.5, 0.5, 0.5],
]
IDS = ["cat", "dog", "bird"]
DOCS = ["Кошка мяукает", "Собака лает", "Птица поёт"]
METAS = [{"animal": "cat"}, {"animal": "dog"}, {"animal": "bird"}]
QUERY_VEC = [0.15, 0.25, 0.35, 0.45]


def run(db, label):
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
    print(f"delete('bird') → count = {db.count()}")


if __name__ == "__main__":
    run(
        ChromaDB("test_col", persist_directory="./chroma_test"),
        "ChromaDB",
    )
    run(
        QdrantDB("test_collection", vector_size=4, in_memory=True),
        "QdrantDB",
    )
    print("\nDONE!\n")
