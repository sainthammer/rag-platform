"""Разбивка текста на чанки для индексации в векторном хранилище.

Публичный API модуля:
    Chunk            — датакласс фрагмента документа (text, metadata, id)
    Chunker          — абстрактный базовый класс для стратегий разбивки
    DocumentLoader   — абстрактный базовый класс для загрузчиков файлов
    FixedSizeChunker — разбивка скользящим окном фиксированного размера
    ByHeaderChunker  — разбивка по Markdown-заголовкам
    SemanticChunker  — объединение параграфов по близости / размеру
    TextLoader       — загрузка .txt-файлов
    MarkdownLoader   — загрузка .md-файлов
    HTMLLoader       — загрузка .html-файлов (требует beautifulsoup4)
    PDFLoader        — загрузка .pdf-файлов (требует pypdf)
    ingest           — единая функция load+chunk для файла с известным расширением
    chunk_text       — низкоуровневая утилита: текст → list[str] (обратная совместимость)
"""

from __future__ import annotations

from .adapters import ByHeaderChunker, FixedSizeChunker, SemanticChunker
from .loaders import HTMLLoader, MarkdownLoader, PDFLoader, TextLoader
from .pipeline import ingest
from .ports import Chunk, Chunker, DocumentLoader

__all__ = [
    "Chunk",
    "Chunker",
    "DocumentLoader",
    "FixedSizeChunker",
    "ByHeaderChunker",
    "SemanticChunker",
    "TextLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "PDFLoader",
    "ingest",
    "chunk_text",
]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Разбить текст на перекрывающиеся фрагменты (низкоуровневая утилита).

    Используется в ``api/routers/ingest.py``. Для нового кода предпочтительнее
    использовать ``FixedSizeChunker`` или ``ingest()``.

    Args:
        text: Исходный текст.
        chunk_size: Максимальный размер чанка в символах.
        chunk_overlap: Перекрытие между соседними чанками в символах.

    Returns:
        Список непустых текстовых фрагментов.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return [c for c in splitter.split_text(text) if c.strip()]
