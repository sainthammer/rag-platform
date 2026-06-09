#!/usr/bin/env python3
"""Демо RAG Platform с реальными сервисами.

Требования:
  • Ollama запущен локально (localhost:11434)
    — модели: llama3.2 (LLM), nomic-embed-text (embeddings)
  • Qdrant запущен в Docker (localhost:6333)
    — docker compose up -d qdrant

Запуск:
    .venv/bin/python demo.py           # все шаги
    .venv/bin/python demo.py --step 3  # конкретный шаг (1-6)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

# ─────────────────────────── ANSI-цвета ──────────────────────────────────────

G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[94m"; R = "\033[91m"; W = "\033[97m"
DIM = "\033[2m"; RESET = "\033[0m"


def header(n: int, title: str) -> None:
    print(f"\n{B}{'─' * 62}{RESET}")
    print(f"{B}  Шаг {n}: {title}{RESET}")
    print(f"{B}{'─' * 62}{RESET}\n")


def ok(msg: str)   -> None: print(f"  {G}✓{RESET}  {msg}")
def info(msg: str) -> None: print(f"  {C}·{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {Y}!{RESET}  {msg}")
def err(msg: str)  -> None: print(f"  {R}✗{RESET}  {msg}", file=sys.stderr)


def show_result(rank: int, doc: str, score: float,
                label: str = "", max_score: float = 1.0) -> None:
    filled = int(min(score / max(max_score, 1e-9), 1.0) * 20)
    tag = f"{DIM}[{label}]{RESET} " if label else ""
    print(f"    {rank}. {tag}{W}{doc[:82]}{RESET}")
    print(f"       {C}{'█' * filled:<20}{RESET}  {Y}{score:.4f}{RESET}")


# ─────────────────────────── конфигурация ────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/v1"
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
EMBED_MODEL  = "nomic-embed-text"
EMBED_DIM    = 768
LLM_MODEL    = "llama3.2"

# Корпус для демо — 10 коротких документов на русском
CORPUS: dict[str, str] = {
    "python":     "Python — высокоуровневый язык программирования для data science и машинного обучения.",
    "docker":     "Docker контейнеры изолируют приложения и их зависимости от хост-системы.",
    "ml":         "Машинное обучение требует большие размеченные датасеты и вычислительные мощности.",
    "api":        "REST API использует HTTP-методы для предоставления доступа к ресурсам по сети.",
    "db":         "Базы данных хранят и извлекают структурированные данные с высокой эффективностью.",
    "qdrant":     "Qdrant — векторная база данных с поддержкой sparse-индекса и гибридного поиска.",
    "chromadb":   "ChromaDB — встраиваемая векторная БД с открытым исходным кодом для RAG-систем.",
    "bm25":       "BM25 — классическая функция ранжирования, основанная на частоте термина и IDF.",
    "rrf":        "Reciprocal Rank Fusion объединяет результаты разных систем поиска через ранги.",
    "embeddings": "Векторные представления текстов (embeddings) кодируют семантику в числовом пространстве.",
}


# ─────────────────────────── фабрики ─────────────────────────────────────────

def make_embed():
    """Создать Ollama nomic-embed-text через OpenAI-совместимый API."""
    from embeddings.adapters import OllamaEmbeddingService
    svc = OllamaEmbeddingService(model=EMBED_MODEL, base_url=OLLAMA_URL, normalize=True)
    return svc.embed


def make_vectorizer():
    from vector_store.bm25 import BM25SparseVectorizer
    v = BM25SparseVectorizer()
    v.fit(list(CORPUS.values()))
    return v


def add_to_store(store, embed_fn, corpus: dict[str, str] | None = None) -> None:
    c = corpus or CORPUS
    ids   = list(c.keys())
    texts = list(c.values())
    store.add(
        ids=ids,
        embeddings=[embed_fn(t) for t in texts],
        documents=texts,
        metadatas=[{"topic": k} for k in ids],
    )


def _clean_qdrant_collection(name: str) -> None:
    """Удалить коллекцию Qdrant, если она существует (для чистого запуска)."""
    try:
        from qdrant_client import QdrantClient
        c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        existing = {col.name for col in c.get_collections().collections}
        if name in existing:
            c.delete_collection(name)
    except Exception:
        pass


# ─────────────────────────── Шаг 1: BM25 ────────────────────────────────────

def step1_bm25() -> None:
    from vector_store.bm25 import BM25SparseVectorizer

    header(1, "BM25SparseVectorizer — токенизация и sparse-вектора")

    v = BM25SparseVectorizer(k1=1.5, b=0.75)
    v.fit(list(CORPUS.values()))

    ok(f"Словарь обучен: {len(v._vocab)} уникальных токенов из {len(CORPUS)} документов")
    info(f"Средняя длина документа (токенов): {v._avgdl:.1f}")

    queries = [
        "машинное обучение датасеты",
        "ранжирование термин частота",      # inflected forms now in vocab
        "векторная база данных sparse",
        "xyzzy quux несуществующее",
    ]
    print()
    for q in queries:
        sv = v.transform(q)
        if not sv.indices:
            warn(f"'{q}' → пустой вектор (OOV)")
        else:
            top = sorted(zip(sv.values, sv.indices), reverse=True)[:4]
            repr_ = ", ".join(f"{list(v._vocab.keys())[i]}:{w:.2f}" for w, i in top)
            ok(f"'{q}' → {len(sv.indices)} термов: [{repr_}]")


# ─────────────────────────── Шаг 2: Dense ChromaDB vs Qdrant ─────────────────

def step2_dense_compare(tmp_path: Path) -> None:
    from vector_store.adapters import ChromaDB, QdrantVectorStore

    header(2, "Dense-поиск: ChromaDB vs Qdrant  [Ollama nomic-embed-text 768d]")
    info(f"Embedding-модель: {EMBED_MODEL}  dim={EMBED_DIM}")

    embed      = make_embed()
    vectorizer = make_vectorizer()

    _clean_qdrant_collection("demo_dense")

    chroma = ChromaDB("demo_dense", persist_directory=str(tmp_path / "chroma"))
    qdrant = QdrantVectorStore(
        "demo_dense", vector_size=EMBED_DIM, vectorizer=vectorizer,
        host=QDRANT_HOST, port=QDRANT_PORT,
    )

    info("Индексируем 10 документов…")
    add_to_store(chroma, embed)
    add_to_store(qdrant, embed)
    ok(f"Chroma: {chroma.count()} doc  |  Qdrant: {qdrant.count()} doc")

    queries = [
        ("python data science",          "python"),
        ("векторная база данных",        "qdrant"),
        ("ранжирование термин частота",  "bm25"),   # раньше давал нули!
        ("контейнеризация зависимости",  "docker"),
    ]

    for q, expected in queries:
        q_emb  = embed(q)
        c_res  = chroma.search(q_emb, n_results=3)
        qq_res = qdrant.search(q_emb, n_results=3)

        print(f"\n  {Y}Запрос:{RESET} \"{q}\"  {DIM}(ожидаем topic={expected}){RESET}")

        print(f"  {C}Chroma{RESET}  (L2-dist ↓):")
        for i, (doc, dist) in enumerate(zip(c_res.documents, c_res.distances), 1):
            tag = "←" if c_res.metadatas[i-1].get("topic") == expected else ""
            show_result(i, doc, 1/(1+dist), f"dist={dist:.3f} {tag}", max_score=1.0)

        print(f"  {C}Qdrant{RESET}  (cosine ↑):")
        max_q = max(qq_res.distances) if qq_res.distances else 1.0
        for i, (doc, score) in enumerate(zip(qq_res.documents, qq_res.distances), 1):
            tag = "←" if qq_res.metadatas[i-1].get("topic") == expected else ""
            show_result(i, doc, score, tag, max_score=max_q)


# ─────────────────────────── Шаг 3: Sparse BM25 vs Dense ────────────────────

def step3_sparse() -> None:
    from vector_store.adapters import QdrantVectorStore

    header(3, "Sparse BM25-поиск vs Dense  [Qdrant Docker]")

    embed      = make_embed()
    vectorizer = make_vectorizer()

    _clean_qdrant_collection("demo_sparse")
    store = QdrantVectorStore(
        "demo_sparse", vector_size=EMBED_DIM, vectorizer=vectorizer,
        host=QDRANT_HOST, port=QDRANT_PORT,
    )
    add_to_store(store, embed)
    ok(f"Проиндексировано: {store.count()} документов")

    cases = [
        ("BM25 ранжирование термин частота IDF",  "точные термины → sparse выиграет"),
        ("python data science",                    "оба должны найти Python-документ"),
        ("контейнер изоляция зависимости",         "dense сработает на синонимах"),
    ]

    for query, note in cases:
        q_emb  = embed(query)
        dense  = store.search(q_emb, n_results=3)
        sparse = store.sparse_search(query, n_results=3)

        print(f"\n  {Y}Запрос:{RESET} \"{query}\"  {DIM}({note}){RESET}")

        max_d = max(dense.distances, default=1.0)
        print(f"  {C}Dense (cosine):{RESET}")
        for i, (doc, score) in enumerate(zip(dense.documents, dense.distances), 1):
            show_result(i, doc, score, max_score=max_d)

        max_s = max(sparse.distances, default=1.0)
        print(f"  {C}Sparse / BM25:{RESET}")
        for i, (doc, score) in enumerate(zip(sparse.documents, sparse.distances), 1):
            show_result(i, doc, score, max_score=max_s)

        d_set = set(dense.documents[:3]); s_set = set(sparse.documents[:3])
        overlap = d_set & s_set
        if overlap:
            ok(f"Пересечение top-3: {len(overlap)} общих")
        diff = s_set - d_set
        if diff:
            for doc in diff:
                info(f"Sparse нашёл эксклюзивно: «{doc[:70]}»")


# ─────────────────────────── Шаг 4: Hybrid RRF ───────────────────────────────

def step4_hybrid() -> None:
    from vector_store.adapters import HybridVectorStore

    header(4, "HybridVectorStore — RRF fusion dense + BM25 sparse  [Qdrant Docker]")

    embed      = make_embed()
    vectorizer = make_vectorizer()

    _clean_qdrant_collection("demo_hybrid")
    store = HybridVectorStore(
        "demo_hybrid", vector_size=EMBED_DIM, vectorizer=vectorizer,
        host=QDRANT_HOST, port=QDRANT_PORT,
    )
    add_to_store(store, embed)
    ok(f"Проиндексировано: {store.count()} документов")

    cases = [
        ("BM25 ранжирование термин IDF",          "keyword-heavy: sparse должен помочь"),
        ("векторная база данных гибридный поиск",  "оба сигнала согласны → top-1 однозначен"),
        ("контейнеризация приложений деплой",      "dense поймает синонимы"),
    ]

    for query, note in cases:
        q_emb  = embed(query)
        dense  = store.search(q_emb, n_results=3)
        hybrid = store.hybrid_search(query, q_emb, n_results=3, rrf_k=60)

        print(f"\n  {Y}Запрос:{RESET} \"{query}\"  {DIM}({note}){RESET}")

        max_d = max(dense.distances, default=1.0)
        print(f"  {C}Dense-only:{RESET}")
        for i, (doc, score) in enumerate(zip(dense.documents, dense.distances), 1):
            show_result(i, doc, score, max_score=max_d)

        max_h = max(hybrid.distances, default=1.0)
        print(f"  {G}Hybrid / RRF:{RESET}")
        for i, (doc, score) in enumerate(zip(hybrid.documents, hybrid.distances), 1):
            show_result(i, doc, score, "rrf", max_score=max_h)

        d_ids = {m.get("topic") for m in dense.metadatas}
        h_ids = {m.get("topic") for m in hybrid.metadatas}
        new   = h_ids - d_ids
        if new:
            ok(f"Hybrid поднял новые документы: {new}")
        else:
            info("Dense и hybrid дали одинаковый top-3")


# ─────────────────────────── Шаг 5: reindex.py ───────────────────────────────

def step5_reindex(tmp_path: Path) -> None:
    import subprocess

    header(5, "reindex.py — сброс коллекции и переиндексация  [Qdrant Docker + Ollama]")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    samples = {
        "python.txt": (
            "Python — язык программирования с динамической типизацией.\n"
            "Широко применяется в data science, веб-разработке и автоматизации.\n"
            "Экосистема включает numpy, pandas, scikit-learn и многие другие библиотеки."
        ),
        "docker.txt": (
            "Docker позволяет упаковать приложение вместе с его зависимостями в контейнер.\n"
            "Контейнеры изолированы от хост-системы и легко переносятся между окружениями.\n"
            "Docker Compose упрощает запуск многоконтейнерных приложений."
        ),
        "rag.txt": (
            "RAG (Retrieval-Augmented Generation) — архитектура, сочетающая поиск и генерацию.\n"
            "Система сначала ищет релевантные документы, затем передаёт их в языковую модель.\n"
            "Это позволяет давать актуальные ответы без дообучения модели."
        ),
    }
    for name, content in samples.items():
        (docs_dir / name).write_text(content, encoding="utf-8")
    ok(f"Создано {len(samples)} документов в {docs_dir}")

    _clean_qdrant_collection("demo_reindex")

    print(f"\n  {C}Запуск reindex.py (Qdrant backend + Ollama embeddings):{RESET}")
    result = subprocess.run(
        [
            sys.executable, "reindex.py",
            str(docs_dir),
            "--backend", "qdrant",
            "--collection", "demo_reindex",
            "--chunk-size", "200",
            "--chunk-overlap", "20",
            "--embedding-provider", "ollama",
            "--ollama-url", OLLAMA_URL,
            "--ollama-model", EMBED_MODEL,
            "--vector-size", str(EMBED_DIM),
            "--batch-size", "10",
        ],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    )
    print(result.stdout)
    if result.returncode != 0:
        warn(f"STDERR:\n{result.stderr}")
        return
    ok("reindex.py завершился успешно")

    from vector_store.adapters import QdrantVectorStore
    vectorizer = make_vectorizer()
    store = QdrantVectorStore(
        "demo_reindex", vector_size=EMBED_DIM, vectorizer=vectorizer,
        host=QDRANT_HOST, port=QDRANT_PORT,
    )
    count = store.count()
    ok(f"Коллекция 'demo_reindex' содержит {count} векторов")

    embed = make_embed()
    res = store.search(embed("RAG поиск языковая модель"), n_results=3)
    print(f"\n  Поиск «RAG поиск языковая модель» (Ollama embeddings):")
    max_s = max(res.distances, default=1.0)
    for i, (doc, score) in enumerate(zip(res.documents, res.distances), 1):
        show_result(i, doc[:90], score, max_score=max_s)


# ─────────────────────────── Шаг 6: Полный RAG-пайплайн ──────────────────────

def step6_rag_pipeline() -> None:
    import asyncio

    from embeddings.adapters import OllamaEmbeddingService
    from llm.adapters import OllamaProvider
    from retrieval.pipeline import RAGPipeline
    from vector_store.adapters import HybridVectorStore

    header(6, "Полный RAG-пайплайн  [Ollama nomic-embed-text + llama3.2 + Qdrant]")

    # ── компоненты ────────────────────────────────────────────────────────────
    embed_svc  = OllamaEmbeddingService(model=EMBED_MODEL, base_url=OLLAMA_URL)
    llm        = OllamaProvider(model=LLM_MODEL, base_url=OLLAMA_URL)
    vectorizer = make_vectorizer()

    _clean_qdrant_collection("demo_rag")
    store = HybridVectorStore(
        "demo_rag", vector_size=EMBED_DIM, vectorizer=vectorizer,
        host=QDRANT_HOST, port=QDRANT_PORT,
    )

    # ── корпус знаний ─────────────────────────────────────────────────────────
    knowledge = {
        "py":  "Python — высокоуровневый язык с динамической типизацией. Используется в data science, веб-разработке и автоматизации. Популярные библиотеки: numpy, pandas, scikit-learn, FastAPI.",
        "rag": "RAG (Retrieval-Augmented Generation) — архитектура, в которой языковая модель дополняется релевантными документами из векторной базы данных. Это позволяет давать актуальные, точные ответы без дообучения модели.",
        "bm25": "BM25 — классический алгоритм ранжирования документов, основанный на TF-IDF. Вычисляет вес термина с учётом частоты в документе, IDF по корпусу и нормализации по длине документа. Параметры: k1 управляет насыщением TF, b — нормализацией.",
        "qdrant": "Qdrant — векторная база данных с поддержкой dense, sparse и hybrid (RRF) поиска. Хранит точки с именованными векторами, поддерживает фильтрацию по payload, on-disk индексы.",
        "hybrid": "Гибридный поиск совмещает dense-семантику и BM25-точность через Reciprocal Rank Fusion (RRF). RRF суммирует обратные ранги: score = Σ 1/(k+rank), обычно k=60. Это нивелирует разницу в масштабах score.",
    }

    info("Индексируем корпус знаний (Ollama embeddings)…")
    add_to_store(store, embed_svc.embed, corpus=knowledge)
    ok(f"Векторизовано {store.count()} документов")

    # ── RAGPipeline ───────────────────────────────────────────────────────────
    pipeline = RAGPipeline(
        embed_fn=embed_svc.embed,
        vector_db_factory=lambda _: store,
        llm=llm,
    )

    questions = [
        "Что такое BM25 и как он ранжирует документы?",
        "Как работает Qdrant для гибридного поиска?",
        "Объясни принцип RAG-архитектуры кратко.",
    ]

    async def run_questions():
        for q in questions:
            print(f"\n  {Y}Вопрос:{RESET} {q}")
            t0 = time.time()
            resp = await pipeline.ask(q, collection="demo_rag")
            elapsed = (time.time() - t0) * 1000

            ok(f"Ответ ({elapsed:.0f} ms, confidence={resp.confidence:.3f}):")
            # Красивый вывод ответа с переносом строк
            lines = resp.answer.strip().split("\n")
            for line in lines:
                print(f"       {line}")

            if resp.sources:
                info(f"Источники ({len(resp.sources)}):")
                for src in resp.sources[:2]:
                    print(f"       {DIM}[score={src.score:.3f}] {src.text[:75]}…{RESET}")

    asyncio.run(run_questions())


# ─────────────────────────── main ────────────────────────────────────────────

STEPS: dict[int, tuple[str, object]] = {
    1: ("BM25SparseVectorizer",               lambda _: step1_bm25()),
    2: ("Dense ChromaDB vs Qdrant [Ollama]",  step2_dense_compare),
    3: ("Sparse BM25 vs Dense [Qdrant]",      lambda _: step3_sparse()),
    4: ("Hybrid RRF [Qdrant]",                lambda _: step4_hybrid()),
    5: ("reindex.py [Qdrant + Ollama]",       step5_reindex),
    6: ("Полный RAG-пайплайн [llama3.2]",     lambda _: step6_rag_pipeline()),
}


def _check_services() -> bool:
    import socket
    ok_flag = True

    for name, host, port in [
        ("Ollama", "localhost", 11434),
        ("Qdrant", QDRANT_HOST, QDRANT_PORT),
    ]:
        try:
            with socket.create_connection((host, port), timeout=2):
                ok(f"{name} доступен ({host}:{port})")
        except OSError:
            err(f"{name} недоступен ({host}:{port}) — часть шагов может упасть")
            ok_flag = False

    return ok_flag


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо RAG Platform с реальными сервисами")
    parser.add_argument(
        "--step", type=int, choices=list(STEPS),
        help="Запустить только этот шаг (1-6)",
    )
    args = parser.parse_args()

    print(f"\n{B}{'═' * 62}{RESET}")
    print(f"{B}   RAG Platform Demo  —  Ollama + Qdrant + BM25 + llama3.2{RESET}")
    print(f"{B}{'═' * 62}{RESET}\n")

    _check_services()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        steps_to_run = [args.step] if args.step else list(STEPS)

        for n in steps_to_run:
            title, fn = STEPS[n]
            try:
                fn(tmp_path)  # type: ignore[call-arg]
            except Exception as exc:
                import traceback
                print(f"\n{R}  Ошибка в шаге {n} ({title}):{RESET} {exc}")
                traceback.print_exc()

    print(f"\n{G}{'═' * 62}{RESET}")
    print(f"{G}  Демо завершено.{RESET}")
    print(f"{G}{'═' * 62}{RESET}\n")


if __name__ == "__main__":
    main()
