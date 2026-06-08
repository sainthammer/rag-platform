"""Функция ingest: загрузка документа и разбивка на чанки.

Связывает DocumentLoader и Chunker в единый пайплайн.
Формат файла определяется по расширению, стратегия — параметром strategy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import ByHeaderChunker, FixedSizeChunker, SemanticChunker
from .loaders import HTMLLoader, MarkdownLoader, PDFLoader, TextLoader
from .ports import Chunk, Chunker, DocumentLoader

_LOADERS: dict[str, type[DocumentLoader]] = {
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".pdf": PDFLoader,
}

_STRATEGIES = {"fixed", "by_header", "semantic"}


def ingest(
    source_path: str | Path,
    strategy: str = "fixed",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Загрузить документ и разбить его на чанки.

    Определяет загрузчик по расширению файла, затем применяет выбранную
    стратегию разбивки. Метаданные ``source`` и ``filename`` добавляются
    автоматически и могут быть дополнены через параметр ``metadata``.

    Args:
        source_path: Путь к документу. Поддерживаемые форматы:
            ``.txt``, ``.md``, ``.markdown``, ``.html``, ``.htm``, ``.pdf``.
        strategy: Стратегия разбивки — ``"fixed"``, ``"by_header"``, ``"semantic"``.
        chunk_size: Максимальный размер чанка в символах.
        chunk_overlap: Перекрытие соседних чанков (применяется для ``fixed``
            и как fallback внутри ``by_header``).
        metadata: Дополнительные метаданные, добавляемые в каждый чанк.

    Returns:
        Список ``Chunk``-объектов в порядке появления в документе.

    Raises:
        ValueError: Если формат файла или стратегия не поддерживаются.
        FileNotFoundError: Если файл не существует.
    """
    path = Path(source_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    ext = path.suffix.lower()
    loader_cls = _LOADERS.get(ext)
    if loader_cls is None:
        supported = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Неподдерживаемый формат файла: {ext!r}. Поддерживаются: {supported}"
        )

    if strategy not in _STRATEGIES:
        supported_strats = ", ".join(sorted(_STRATEGIES))
        raise ValueError(
            f"Неизвестная стратегия: {strategy!r}. Доступны: {supported_strats}"
        )

    base_meta: dict[str, Any] = {
        "source": str(path),
        "filename": path.name,
        **(metadata or {}),
    }

    loader: DocumentLoader = loader_cls()
    text = loader.load(path)

    chunker: Chunker
    if strategy == "fixed":
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "by_header":
        chunker = ByHeaderChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        chunker = SemanticChunker(chunk_size=chunk_size)

    return chunker.split(text, base_meta)
