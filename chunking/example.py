"""Пример запуска модуля chunking.

Запуск:
    python -m chunking.example
    python -m chunking.example --source README.md --strategy by_header
    python -m chunking.example --source docs/example.pdf --strategy fixed

По умолчанию пример использует встроенный Markdown-текст и прогоняет его через
все стратегии чанкинга. При передаче ``--source`` запускается полный ingestion
pipeline с загрузкой документа с диска.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chunking import Chunk, build_chunker, ingest

SAMPLE_MARKDOWN = """# RAG pipeline

RAG состоит из загрузки документов, разбиения на чанки, построения embedding-векторов
и поиска по vector store.

## Chunking

Chunking делит большой документ на небольшие фрагменты. Это помогает embedding-модели
работать с локальным контекстом и повышает качество поиска.

## Retrieval

Retrieval превращает пользовательский вопрос в embedding-вектор и ищет ближайшие
чанки в vector store.
"""


def main() -> None:
    """Запустить демонстрацию загрузки документа и стратегий чанкинга.

    Returns:
        ``None``. Результаты выводятся в консоль.

    Raises:
        FileNotFoundError: Если переданный через ``--source`` файл не существует.
        ValueError: Если стратегия чанкинга или тип документа не поддерживаются.
        OSError: Если исходный файл нельзя прочитать.
    """
    configure_stdout()
    args = parse_args()

    if args.source:
        run_for_source(
            source=Path(args.source),
            strategies=[args.strategy],
            chunk_size=args.chunk_size,
            limit=args.limit,
            preview_chars=args.preview_chars,
        )
        return

    run_for_text(
        text=SAMPLE_MARKDOWN,
        strategies=["fixed", "by_header", "semantic"],
        chunk_size=args.chunk_size,
        limit=args.limit,
        preview_chars=args.preview_chars,
    )


def parse_args() -> argparse.Namespace:
    """Прочитать аргументы командной строки.

    Returns:
        Пространство имен с путем к источнику, стратегией и размером чанка.
    """
    parser = argparse.ArgumentParser(description="Демонстрация модуля chunking")
    parser.add_argument(
        "--source",
        help="Путь к документу. Если не задан, используется встроенный Markdown-пример.",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "by_header", "semantic"],
        default="by_header",
        help="Стратегия чанкинга для документа, переданного через --source.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=140,
        help="Максимальный размер чанка в символах.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Сколько чанков показывать для каждой стратегии.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=90,
        help="Сколько символов показывать из текста чанка. 0 означает показать весь чанк.",
    )
    return parser.parse_args()


def configure_stdout() -> None:
    """Настроить консольный вывод для Unicode-текста на Windows.

    Returns:
        ``None``. Поток ``stdout`` перенастраивается на UTF-8, если окружение
        поддерживает ``reconfigure``.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_for_source(
    source: Path,
    strategies: list[str],
    chunk_size: int,
    limit: int,
    preview_chars: int,
) -> None:
    """Запустить одну или несколько стратегий для документа.

    Args:
        source: Путь к документу.
        strategies: Список стратегий чанкинга.
        chunk_size: Максимальный размер чанка в символах.
        limit: Максимальное количество чанков для вывода.
        preview_chars: Максимальная длина текстового превью. 0 показывает весь чанк.

    Returns:
        ``None``. Сводка по чанкам выводится в консоль.

    Raises:
        FileNotFoundError: Если документ не существует.
        ValueError: Если стратегия или тип документа не поддерживается.
        OSError: Если документ нельзя прочитать.
    """
    print(f"Источник: {source}")
    print(f"Размер чанка: {chunk_size}")

    for strategy in strategies:
        chunks = ingest(source, strategy=strategy, chunk_size=chunk_size)
        print(f"\nСтратегия: {strategy}")
        print(f"Количество чанков: {len(chunks)}")
        print_chunks(chunks, limit=limit, preview_chars=preview_chars)

def run_for_text(
    text: str,
    strategies: list[str],
    chunk_size: int,
    limit: int,
    preview_chars: int,
) -> None:
    """Запустить стратегии чанкинга для встроенного текста без файловой системы.

    Args:
        text: Исходный текст.
        strategies: Список стратегий чанкинга.
        chunk_size: Максимальный размер чанка в символах.
        limit: Максимальное количество чанков для вывода.
        preview_chars: Максимальная длина текстового превью. 0 показывает весь чанк.

    Returns:
        ``None``. Сводка по чанкам выводится в консоль.

    Raises:
        ValueError: Если стратегия чанкинга не поддерживается.
    """
    metadata = {
        "source": "embedded-sample.md",
        "source_name": "embedded-sample.md",
        "content_type": "text/markdown",
    }

    print("Источник: встроенный Markdown-пример")
    print(f"Размер чанка: {chunk_size}")

    for strategy in strategies:
        chunker = build_chunker(strategy, chunk_size=chunk_size)
        chunks = chunker.chunk(text, metadata)
        print(f"\nСтратегия: {strategy}")
        print(f"Количество чанков: {len(chunks)}")
        print_chunks(chunks, limit=limit, preview_chars=preview_chars)


def print_chunks(chunks: list[Chunk], preview_chars: int = 90, limit: int = 10) -> None:
    """Вывести краткую информацию о чанках.

    Args:
        chunks: Чанки, которые нужно показать.
        preview_chars: Максимальная длина текстового превью. 0 показывает весь чанк.
        limit: Максимальное количество чанков для вывода.

    Returns:
        ``None``. Данные выводятся в консоль.
    """
    shown_chunks = chunks[: max(0, limit)]
    for chunk in shown_chunks:
        normalized_text = " ".join(chunk.text.split())
        preview = normalized_text if preview_chars <= 0 else normalized_text[:preview_chars]
        section = chunk.metadata.get("section")
        section_text = f", section={section!r}" if section else ""
        print(
            "  "
            f"id={chunk.id}, "
            f"index={chunk.metadata.get('chunk_index')}, "
            f"strategy={chunk.metadata.get('chunk_strategy')}"
            f"{section_text}"
        )
        print(f"    {preview}")

    hidden_count = len(chunks) - len(shown_chunks)
    if hidden_count > 0:
        print(
            f"  ... еще {hidden_count} чанков скрыто. "
            "Используйте --limit, чтобы показать больше."
        )


if __name__ == "__main__":
    main()
