"""Сравнение chunk_size=256 vs 512 по метрикам качества RAG-retrieval.

Скрипт индексирует выборку документов из docs/ в две отдельные коллекции
с разными размерами чанков, затем прогоняет тестовые вопросы и сравнивает:
  - Количество проиндексированных чанков
  - Среднюю длину чанка
  - Среднее значение confidence (max retrieval score)
  - Среднюю задержку retrieval (embed + search), мс

Запуск (внутри Docker):
    docker compose exec api python compare_chunks.py
    docker compose exec api python compare_chunks.py --max-files 50 --max-questions 10
    docker compose exec api python compare_chunks.py --output results/chunk_comparison.json

Запуск локально (нужен запущенный ChromaDB):
    PYTHONPATH=. python compare_chunks.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Тестовые вопросы
# ---------------------------------------------------------------------------

_DEFAULT_QUESTIONS = [
    "Что такое PEP 8 и для чего он нужен?",
    "Что такое декоратор в Python?",
    "Как работает garbage collector в Python?",
    "Что такое list comprehension?",
    "Какие типы данных есть в Python?",
    "Что такое Django ORM?",
    "Как работает QuerySet в Django?",
    "Что такое Signal в Django?",
    "Что такое HTTP методы?",
    "Что такое замыкание в Python?",
    "Что такое генератор в Python?",
    "Чем отличается list от tuple?",
    "Что такое MVC паттерн?",
    "Как работает async/await в Python?",
    "Что такое lambda-функция?",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

async def _seed_collection(
    collection_name: str,
    chunk_size: int,
    docs_dir: str,
    max_files: int,
    embed_service,
    settings,
) -> int:
    """Проиндексировать docs/ в коллекцию с заданным chunk_size.

    Возвращает количество проиндексированных чанков.
    """
    from chunking import ingest as chunking_ingest
    from config import build_vector_db

    docs_path = Path(docs_dir)
    md_files = sorted(docs_path.rglob("*.md"))[:max_files]
    if not md_files:
        raise FileNotFoundError(f"Нет .md файлов в {docs_dir}")

    db = build_vector_db(settings, collection=collection_name)

    # Сбрасываем коллекцию перед заполнением
    try:
        db.client.delete_collection(collection_name)
        db = build_vector_db(settings, collection=collection_name)
        print(f"  Коллекция '{collection_name}' очищена")
    except Exception:
        pass

    print(f"  Чанкинг {len(md_files)} файлов (chunk_size={chunk_size})…")

    # Параллельный чанкинг батчами по 20 файлов
    batch_size = 20
    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []

    for i in range(0, len(md_files), batch_size):
        batch = md_files[i : i + batch_size]
        results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    chunking_ingest, p,
                    strategy="by_header",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_size // 5,
                )
                for p in batch
            ],
            return_exceptions=True,
        )
        for path, result in zip(batch, results):
            if isinstance(result, Exception):
                continue
            rel = path.relative_to(docs_path).as_posix()
            for idx, chunk in enumerate(result):
                all_texts.append(chunk.text)
                all_ids.append(f"{rel}_chunk{idx}")
                all_metas.append(chunk.metadata)

    if not all_texts:
        return 0

    print(f"  Embedding {len(all_texts)} чанков…")
    embeddings = await asyncio.to_thread(embed_service.embed_batch, all_texts)
    await asyncio.to_thread(db.add, all_ids, embeddings, all_texts, all_metas)
    return len(all_texts)


async def _benchmark_retrieval(
    collection_name: str,
    questions: list[str],
    embed_fn,
    settings,
    n_results: int = 5,
) -> dict:
    """Прогнать вопросы через retrieval и вернуть агрегированные метрики."""
    from config import build_vector_db

    db = build_vector_db(settings, collection=collection_name)

    confidences: list[float] = []
    latencies: list[float] = []
    chunk_lens: list[float] = []
    source_counts: list[int] = []

    for question in questions:
        t0 = time.monotonic()
        embedding = await asyncio.to_thread(embed_fn, question)
        result = await asyncio.to_thread(db.search, embedding, n_results)
        latency_ms = (time.monotonic() - t0) * 1000

        if result.documents:
            scores = [1.0 / (1.0 + d) for d in result.distances]
            confidences.append(max(scores))
            source_counts.append(len(result.documents))
            chunk_lens.extend(len(doc) for doc in result.documents)
        else:
            confidences.append(0.0)
            source_counts.append(0)

        latencies.append(latency_ms)

    return {
        "avg_confidence": statistics.mean(confidences) if confidences else 0.0,
        "avg_latency_ms": statistics.mean(latencies),
        "avg_source_count": statistics.mean(source_counts) if source_counts else 0.0,
        "avg_chunk_len": statistics.mean(chunk_lens) if chunk_lens else 0.0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
    }


# ---------------------------------------------------------------------------
# Вывод результатов
# ---------------------------------------------------------------------------

def _print_table(results: dict) -> None:
    col_w = 22
    metric_w = 36

    header = f"{'Метрика':<{metric_w}}" + "".join(
        f"{f'chunk_size={cs}':^{col_w}}" for cs in sorted(results)
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    metrics = [
        ("Чанков проиндексировано", "total_chunks", "{:.0f}"),
        ("Средняя длина чанка, симв.", "avg_chunk_len", "{:.0f}"),
        ("Средний confidence", "avg_confidence", "{:.4f}"),
        ("Среднее кол-во источников", "avg_source_count", "{:.2f}"),
        ("Средняя latency, мс", "avg_latency_ms", "{:.1f}"),
        ("P95 latency, мс", "p95_latency_ms", "{:.1f}"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<{metric_w}}"
        values = [results[cs].get(key, 0) for cs in sorted(results)]
        row += "".join(f"{fmt.format(v):^{col_w}}" for v in values)

        # Подсветить лучшее значение звёздочкой
        if key == "avg_confidence":
            best = max(values)
            row += f"  ← лучше: chunk_size={sorted(results)[values.index(best)]}" if len(set(f"{v:.4f}" for v in values)) > 1 else ""
        elif key in ("avg_latency_ms", "p95_latency_ms"):
            best = min(values)
            row += f"  ← быстрее: chunk_size={sorted(results)[values.index(best)]}" if len(set(f"{v:.1f}" for v in values)) > 1 else ""

        print(row)

    print("─" * len(header))


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from config import build_embedding_service, settings

    questions = _DEFAULT_QUESTIONS[: args.max_questions]
    chunk_sizes = [256, 512]
    results: dict[int, dict] = {}

    print(f"\n{'━'*60}")
    print(f"  Сравнение chunk_size: {chunk_sizes}")
    print(f"  Файлов для индексации : {args.max_files}")
    print(f"  Тестовых вопросов     : {len(questions)}")
    print(f"  Docs dir              : {args.docs_dir}")
    print(f"{'━'*60}\n")

    embed_service = await asyncio.to_thread(build_embedding_service, settings)

    for chunk_size in chunk_sizes:
        collection = f"bench_{chunk_size}"
        print(f"[chunk_size={chunk_size}] Индексация → '{collection}'")

        total_chunks = await _seed_collection(
            collection_name=collection,
            chunk_size=chunk_size,
            docs_dir=args.docs_dir,
            max_files=args.max_files,
            embed_service=embed_service,
            settings=settings,
        )
        print(f"  Проиндексировано: {total_chunks} чанков\n")

        print(f"[chunk_size={chunk_size}] Retrieval benchmark…")
        metrics = await _benchmark_retrieval(
            collection_name=collection,
            questions=questions,
            embed_fn=embed_service.embed,
            settings=settings,
        )
        metrics["total_chunks"] = total_chunks
        results[chunk_size] = metrics
        print(f"  avg_confidence={metrics['avg_confidence']:.4f}  avg_latency={metrics['avg_latency_ms']:.1f}ms\n")

    _print_table(results)

    # Сохраняем JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "max_files": args.max_files,
            "max_questions": args.max_questions,
            "docs_dir": args.docs_dir,
            "questions": questions,
        },
        "results": {str(cs): v for cs, v in results.items()},
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nРезультаты сохранены → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение chunk_size=256 vs 512")
    parser.add_argument("--max-files", type=int, default=100,
                        help="Максимальное число .md файлов для индексации (default: 100)")
    parser.add_argument("--max-questions", type=int, default=15,
                        help="Число тестовых вопросов (default: 15)")
    parser.add_argument("--docs-dir", default="docs",
                        help="Директория с документами (default: docs)")
    parser.add_argument("--output", default="results/chunk_comparison.json",
                        help="Путь для сохранения JSON-результатов")
    args = parser.parse_args()
    asyncio.run(main(args))
