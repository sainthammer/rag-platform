"""CLI для разбивки документов на чанки и опциональной индексации.

Использование::

    python -m chunking.ingest --source doc.pdf --strategy fixed --chunk_size 512
    python -m chunking.ingest --source docs/guide.md --strategy by_header
    python -m chunking.ingest --source notes.txt --strategy semantic --collection my_docs

Аргументы:
    --source        Путь к документу (.txt, .md, .html, .pdf)
    --strategy      Стратегия разбивки: fixed | by_header | semantic  (default: fixed)
    --chunk_size    Максимальный размер чанка в символах  (default: 512)
    --chunk_overlap Перекрытие соседних чанков  (default: 50)
    --collection    Если задан — индексирует в ChromaDB с этим именем коллекции
    --chroma_path   Путь к директории ChromaDB  (default: .chroma)
    --output        Если задан — сохраняет чанки в JSON-файл
    --quiet         Не выводить подробности, только итоговую строку
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI.

    Args:
        argv: Список аргументов командной строки. ``None`` → ``sys.argv[1:]``.

    Returns:
        Код завершения: 0 — успех, 1 — ошибка.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"Ошибка: файл не найден: {source}", file=sys.stderr)
        return 1

    # --- Разбивка ---
    from chunking.pipeline import ingest  # noqa: PLC0415

    try:
        chunks = ingest(
            source_path=source,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except ValueError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1

    if not chunks:
        print("Предупреждение: документ не содержит текста или все чанки пусты.")
        return 0

    if not args.quiet:
        _print_summary(chunks, args)

    # --- Запись в JSON ---
    if args.output:
        out = Path(args.output)
        data = [
            {"id": c.id, "text": c.text, "metadata": c.metadata}
            for c in chunks
        ]
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"\nЧанки сохранены: {out}")

    # --- Индексация в ChromaDB ---
    if args.collection:
        _index_to_chroma(chunks, args)

    # Итог — всегда печатаем даже в quiet-режиме
    print(f"Готово: {len(chunks)} чанков из {source.name}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chunking.ingest",
        description="Разбить документ на чанки и (опционально) проиндексировать в ChromaDB.",
    )
    parser.add_argument("--source", required=True, help="Путь к файлу")
    parser.add_argument(
        "--strategy",
        default="fixed",
        choices=["fixed", "by_header", "semantic"],
        help="Стратегия разбивки (default: fixed)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512, metavar="N",
        help="Максимальный размер чанка в символах (default: 512)",
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=50, metavar="N",
        help="Перекрытие соседних чанков в символах (default: 50)",
    )
    parser.add_argument(
        "--collection", default=None, metavar="NAME",
        help="Имя коллекции ChromaDB для индексации",
    )
    parser.add_argument(
        "--chroma_path", default=".chroma", metavar="PATH",
        help="Директория ChromaDB (default: .chroma)",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Сохранить чанки в JSON-файл",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Только итоговая строка без подробностей",
    )
    return parser


def _print_summary(chunks, args) -> None:
    """Напечатать краткий отчёт о разбивке."""
    from chunking.ports import Chunk  # noqa: PLC0415

    total_chars = sum(len(c.text) for c in chunks)
    avg_chars = total_chars // len(chunks)

    print(f"\n{'-' * 60}")
    print(f"  Файл:     {args.source}")
    print(f"  Стратегия: {args.strategy}  chunk_size={args.chunk_size}  overlap={args.chunk_overlap}")
    print(f"  Чанков:   {len(chunks)}  (всего {total_chars} симв., ср. {avg_chars} симв.)")
    print(f"{'-' * 60}")

    # Предпросмотр первых трёх чанков
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.text[:80].replace("\n", " ")
        ellipsis = "…" if len(chunk.text) > 80 else ""
        print(f"  [{i}] {preview}{ellipsis}")

    if len(chunks) > 3:
        print(f"  … и ещё {len(chunks) - 3} чанков")


def _index_to_chroma(chunks, args) -> None:
    """Проиндексировать чанки в ChromaDB через FakeEmbeddingService.

    В production замените FakeEmbeddingService на реальный сервис
    через ``build_embedding_service(settings)``.
    """
    from embeddings.adapters import FakeEmbeddingService  # noqa: PLC0415
    from vector_store.adapters import ChromaDB  # noqa: PLC0415

    if not args.quiet:
        print(f"\nИндексация в ChromaDB (коллекция: {args.collection!r}) …")

    embed_service = FakeEmbeddingService(size=128, normalize=True)
    db = ChromaDB(args.collection, persist_directory=args.chroma_path)

    embeddings = embed_service.embed_batch([c.text for c in chunks])
    db.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    if not args.quiet:
        print(f"  Проиндексировано {len(chunks)} чанков → {args.chroma_path}/{args.collection}")


if __name__ == "__main__":
    sys.exit(main())
