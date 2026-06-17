"""Ingestion pipeline: загрузить документ, разбить его и вернуть чанки."""

from __future__ import annotations

from pathlib import Path

from .chunkers import build_chunker
from .dataclasses import Chunk
from .loaders import get_loader
from .ports import Chunker, DocumentLoader


def ingest(
    source_path: str | Path,
    strategy: str | Chunker = "fixed",
    chunk_size: int = 1000,
    loader: DocumentLoader | None = None,
) -> list[Chunk]:
    """Загрузить документ с диска и разбить его на чанки.

    Args:
        source_path: Локальный путь к PDF, Markdown, HTML или текстовому документу.
        strategy: Название стратегии чанкинга: ``fixed``, ``by_header`` или
            ``semantic``. Также можно передать готовый объект ``Chunker``.
        chunk_size: Целевой максимальный размер чанка в символах.
        loader: Опциональный загрузчик для тестов или кастомных форматов.

    Returns:
        Упорядоченный список чанков со стабильными id и метаданными источника.

    Raises:
        FileNotFoundError: Если документ не существует.
        ValueError: Если путь указывает не на файл, тип документа не поддержан
            или стратегия чанкинга неизвестна.
        OSError: Если документ нельзя прочитать.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Документ не существует: {path}")
    if not path.is_file():
        raise ValueError(f"Источник документа должен быть файлом: {path}")

    document_loader = loader or get_loader(path)
    text, metadata = document_loader.load(path)
    chunker = build_chunker(strategy, chunk_size=chunk_size)
    return chunker.chunk(text, metadata)
