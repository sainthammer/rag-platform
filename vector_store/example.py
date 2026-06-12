"""
Примеры использования модуля vector_store.

Каждый пример — отдельная функция, которую можно читать сверху вниз:
    1. подготовка данных
    2. действие
    3. что ожидаем увидеть и почему
    4. проверка (assert), что результат совпал с ожиданием

Можно использовать как смоук-тест: просто запустить файл.
Если всё хорошо — в конце напечатает DONE, если нет — упадёт с AssertionError.
"""

from vector_store.adapters import ChromaDB, HybridVectorStore, QdrantDB
from vector_store.ports import VectorDB
from vector_store.reindex import reindex
from vector_store.store_dataclasses import MetadataFilter, SearchResult
from vector_store.utils import reciprocal_rank_fusion

# ----------------- Тестовые данные -----------------
#
# Три "документа" про животных. Эмбеддинги 4-мерные, придуманы вручную так,
# чтобы было видно геометрию:
#   - вектор запроса QUERY_VEC почти совпадает с вектором "cat"
#   - значит во всех примерах поиска ожидаем top-1 = "cat"

IDS = ["cat", "dog", "bird"]

DOCS = ["Кошка мяукает", "Собака лает", "Птица поёт"]

EMBEDDINGS = [
    [0.1, 0.2, 0.3, 0.4],  # cat
    [0.9, 0.8, 0.7, 0.6],  # dog
    [0.5, 0.5, 0.5, 0.5],  # bird
]

METAS = [
    {
        "animal": "cat",
        "language": "ru",
        "page_num": 1,
        "source_file": "docs/cats.pdf",
        "section": "введение",
    },
    {
        "animal": "dog",
        "language": "ru",
        "page_num": 2,
        "source_file": "docs/dogs.pdf",
        "section": "глава 1",
    },
    {
        "animal": "bird",
        "language": "en",
        "page_num": 1,
        "source_file": "docs/birds.pdf",
        "section": "введение",
    },
]

QUERY_VEC = [0.15, 0.25, 0.35, 0.45]  # почти "cat"


# ----------------- Помощники для вывода -----------------


def header(title: str) -> None:
    print(f"\n{'═' * 50}")
    print(f"  {title}")
    print(f"{'═' * 50}")


def step(text: str) -> None:
    print(f"\n>>> {text}")


def show_results(result: SearchResult) -> None:
    """Печатает результат поиска в виде таблички."""
    if len(result.ids) == 0:
        print("    (пусто)")
        return
    for i in range(len(result.ids)):
        dist = result.distances[i] if result.distances else float("nan")
        print(f"    [{i + 1}] id={result.ids[i]!r:8} dist={dist:.4f} doc={result.documents[i]!r}")


# ----------------- Примеры -----------------


def example_crud(db: VectorDB, label: str) -> None:
    """
    Базовый цикл работы с коллекцией: add -> count -> search -> delete.
    Одинаково работает для ChromaDB и QdrantDB.
    """
    header(f"Базовый CRUD: {label}")

    step("add(): кладём 3 документа (cat, dog, bird)")
    db.add(IDS, EMBEDDINGS, DOCS, METAS)
    print(f"    count() = {db.count()}")
    assert db.count() == 3, "после add должно быть ровно 3 записи"

    step("search(): ищем 2 ближайших к запросу, похожему на 'cat'")
    result = db.search(QUERY_VEC, n_results=2)
    show_results(result)
    print("    ожидание: top-1 = cat (вектор запроса почти совпадает с ним)")
    assert "Кошка" in result.documents[0], "top-1 должен быть документ про кошку"

    step("delete(): удаляем cat и bird")
    db.delete(["cat", "bird"])
    print(f"    count() = {db.count()}")
    assert db.count() == 1, "после удаления двух записей должна остаться одна"
    print("    осталась только запись dog — как и ожидали")


def example_metadata_filters(db: ChromaDB) -> None:
    """
    Поиск с фильтрами по метаданным (MetadataFilter) в ChromaDB.
    Показывает три случая: один фильтр, несколько фильтров (объединяются
    через AND), и оператор in.
    """
    header("Поиск с фильтрами по метаданным: ChromaDB")

    db.add(IDS, EMBEDDINGS, DOCS, METAS)
    print(f"в коллекции {db.count()} записей: cat(ru), dog(ru), bird(en)")

    step("один фильтр: language == 'ru'")
    result = db.search(QUERY_VEC, n_results=3, filters=[MetadataFilter("language", "eq", "ru")])
    show_results(result)
    print("    ожидание: только cat и dog, у bird язык 'en'")
    print("    примечание: id в выводе — UUID, потому что add() прогоняет их через to_uuid")
    assert set(result.documents) == {"Кошка мяукает", "Собака лает"}, (
        "фильтр language=ru должен оставить cat и dog"
    )

    step("несколько фильтров (AND): language=='ru' И page_num>=2 И source_file=='docs/dogs.pdf'")
    result = db.search(
        QUERY_VEC,
        n_results=3,
        filters=[
            MetadataFilter("language", "eq", "ru"),
            MetadataFilter("page_num", "gte", 2),
            MetadataFilter("source_file", "eq", "docs/dogs.pdf"),
        ],
    )
    show_results(result)
    print("    ожидание: только dog — единственный, кто проходит все три условия")
    assert result.documents == ["Собака лает"], "под все три условия подходит только dog"

    step("оператор in: section входит в ['введение', 'глава 1']")
    result = db.search(
        QUERY_VEC,
        n_results=3,
        filters=[MetadataFilter("section", "in", ["введение", "глава 1"])],
    )
    show_results(result)
    print("    ожидание: все три записи — каждая попадает в один из разделов")
    assert set(result.documents) == set(DOCS), "под фильтр in подходят все записи"


def example_rrf() -> None:
    """
    reciprocal_rank_fusion: как объединяются два ранжированных списка.
    RRF даёт документу очки за позицию в каждом списке: score = sum(1 / (k + rank)).
    Чем выше документ в обоих списках — тем больше итоговый score.
    """
    header("Reciprocal Rank Fusion (RRF)")

    step("документ 'b' стоит первым в обоих списках -> должен победить")
    dense_ranking = ["b", "a", "c"]  # как будто результат dense-поиска
    sparse_ranking = ["b", "a", "d"]  # как будто результат sparse-поиска
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    print(f"    вход:  dense={dense_ranking}, sparse={sparse_ranking}")
    print(f"    выход: {fused}")
    assert fused[0][0] == "b", "'b' первый в обоих списках и должен победить"

    step("документы из обоих списков попадают в результат, даже если встречаются один раз")
    fused = reciprocal_rank_fusion([["x"], ["y", "z"]])
    print(f"    выход: {fused}")
    assert {doc_id for doc_id, _ in fused} == {"x", "y", "z"}

    step("крайние случаи: пустой вход не падает")
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    print("    ok")

    step("проверка формулы: единственный документ на позиции 0 при k=60")
    fused = reciprocal_rank_fusion([["only"]], k=60)
    print(f"    score = {fused[0][1]:.6f}, ожидаем 1/(60+1) = {1 / 61:.6f}")
    assert abs(fused[0][1] - 1 / 61) < 1e-9, "score должен считаться как 1 / (k + rank + 1)"


def example_hybrid_search() -> None:
    """
    HybridVectorStore: гибридный поиск = dense (по векторам) + sparse (по словам),
    результаты объединяются через RRF.

    Данные подобраны так, что документ про питон должен победить дважды:
      - его вектор совпадает с вектором запроса (dense)
      - в тексте запроса есть слова "питон" и "язык" (sparse)
    """
    header("Гибридный поиск: HybridVectorStore")

    ids = ["doc-1", "doc-2", "doc-3"]
    documents = [
        "кошки любят спать на солнце",
        "питон это язык программирования",
        "qdrant хранит векторы",
    ]
    # ортогональные вектора: каждый документ "смотрит" в свою сторону
    embeddings = [
        [1.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1.0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1.0, 0, 0, 0, 0, 0],
    ]
    metadatas = [{"topic": "cats"}, {"topic": "python"}, {"topic": "db"}]

    step("создаём QdrantDB с use_sparse=True и оборачиваем в HybridVectorStore")
    store = QdrantDB(collection="hybrid_smoke", vector_size=8, in_memory=True, use_sparse=True)
    hybrid = HybridVectorStore(store)
    hybrid.add(ids, embeddings, documents, metadatas)
    print(f"    count() = {hybrid.count()}")

    step("ищем: вектор запроса = вектор doc-2, текст запроса = 'питон язык'")
    query_embedding = [0, 1.0, 0, 0, 0, 0, 0, 0]  # совпадает с doc-2
    result = hybrid.search(query_embedding=query_embedding, query_text="питон язык", n_results=3)
    show_results(result)
    print("    ожидание: top-1 = doc-2, он выигрывает и dense, и sparse")
    print("    примечание: в distances лежит RRF-score, больше = лучше")
    assert result.documents[0] == "питон это язык программирования"
    assert result.distances == sorted(result.distances, reverse=True), "score должны убывать"

    step("гибридному поиску обязательно нужен query_text")
    try:
        hybrid.search(query_embedding=query_embedding, query_text=None)
        raise AssertionError("ожидали ValueError, но поиск не упал")
    except ValueError as error:
        print(f"    без query_text получаем ValueError: {error}")


def example_reindex() -> None:
    """
    reindex: миграция коллекции из Chroma в Qdrant с пересчётом эмбеддингов.

    Сценарий: документы лежали в Chroma, мы хотим перенести их в Qdrant
    со sparse-индексом, чтобы был доступен гибридный поиск.
    Эмбеддинги пересчитываются заново через embed_fn — поэтому модель
    в source и target может быть разной.
    """
    header("Переиндексация: Chroma -> Qdrant")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    step("источник: коллекция в Chroma (наполняем, если пустая)")
    source = ChromaDB(collection="docs_old", persist_directory="./chroma_data")
    if source.count() == 0:
        docs = [
            "Qdrant — векторная база данных на Rust",
            "ChromaDB — простая embedded-векторная БД",
            "RRF объединяет результаты dense и sparse поиска",
        ]
        source.add(
            ids=["doc-1", "doc-2", "doc-3"],
            embeddings=embed_fn(docs),
            documents=docs,
            metadatas=[{"lang": "ru"} for _ in docs],
        )
    print(f"    в source {source.count()} записей")

    step("цель: новая коллекция в Qdrant со sparse-индексом")
    target = QdrantDB(
        collection="docs_v2",
        vector_size=model.get_sentence_embedding_dimension(),
        in_memory=True,
        use_sparse=True,
    )

    step("reindex(): читаем из source, пересчитываем вектора, пишем в target")
    n = reindex(source=source, target=target, embed_fn=embed_fn, batch_size=128)
    print(f"    переиндексировано: {n}")
    assert target.count() == source.count(), (
        "в target должно быть столько же записей, сколько в source"
    )

    step("проверяем результат гибридным поиском по новой коллекции")
    hybrid = HybridVectorStore(store=target, k=60)
    query = "гибридный поиск с RRF"
    result = hybrid.search(
        query_embedding=embed_fn([query])[0],
        n_results=2,
        query_text=query,
    )
    show_results(result)
    print("    ожидание: top-1 = документ про RRF")
    assert "RRF" in result.documents[0], "по запросу про RRF первым должен прийти документ про RRF"


if __name__ == "__main__":
    example_crud(ChromaDB("test_col", persist_directory="./chroma_test"), "ChromaDB")
    example_crud(QdrantDB("test_collection", vector_size=4, in_memory=True), "QdrantDB")
    example_metadata_filters(ChromaDB("test_filters", persist_directory="./chroma_filters"))
    example_rrf()
    example_hybrid_search()
    example_reindex()
    print("\nDONE! Все примеры отработали, все проверки прошли.\n")
