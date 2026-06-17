"""Публичный API для загрузки документов, чанкинга и ingestion pipeline."""

from .chunkers import ByHeaderChunker, FixedSizeChunker, SemanticChunker, build_chunker
from .dataclasses import Chunk
from .loaders import HTMLLoader, MarkdownLoader, PDFLoader, TextLoader, get_loader
from .pipeline import ingest
from .ports import Chunker, DocumentLoader

__all__ = [
    "ByHeaderChunker",
    "Chunk",
    "Chunker",
    "DocumentLoader",
    "FixedSizeChunker",
    "HTMLLoader",
    "MarkdownLoader",
    "PDFLoader",
    "SemanticChunker",
    "TextLoader",
    "build_chunker",
    "get_loader",
    "ingest",
]
