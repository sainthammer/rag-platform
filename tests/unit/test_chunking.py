"""Unit-тесты для модуля chunking.

Покрывают:
  - Chunk dataclass
  - FixedSizeChunker: размер, метаданные, уникальность id
  - ByHeaderChunker: разбивка по заголовкам, fallback без заголовков
  - SemanticChunker: размерный режим, режим с embed_fn
  - TextLoader, MarkdownLoader, HTMLLoader, PDFLoader
  - ingest(): полный цикл load+chunk, ошибки на неверные аргументы
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from chunking import (
    ByHeaderChunker,
    Chunk,
    FixedSizeChunker,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    SemanticChunker,
    TextLoader,
    ingest,
)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

def test_chunk_defaults() -> None:
    chunk = Chunk(text="hello")
    assert chunk.text == "hello"
    assert chunk.metadata == {}
    assert chunk.id == ""


def test_chunk_with_all_fields() -> None:
    chunk = Chunk(text="hi", metadata={"source": "doc.txt"}, id="c_001")
    assert chunk.metadata["source"] == "doc.txt"
    assert chunk.id == "c_001"


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

_LONG_TEXT = " ".join(["слово"] * 300)  # ~1500 символов


def test_fixed_chunker_splits_long_text() -> None:
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split(_LONG_TEXT)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 220  # небольшой запас из-за перекрытия


def test_fixed_chunker_short_text_single_chunk() -> None:
    chunker = FixedSizeChunker(chunk_size=500)
    chunks = chunker.split("короткий текст")
    assert len(chunks) == 1
    assert chunks[0].text == "короткий текст"


def test_fixed_chunker_metadata_inherited() -> None:
    chunker = FixedSizeChunker(chunk_size=200)
    chunks = chunker.split(_LONG_TEXT, metadata={"doc_id": "test_doc"})
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == "test_doc"
        assert chunk.metadata["strategy"] == "fixed"


def test_fixed_chunker_ids_unique() -> None:
    chunker = FixedSizeChunker(chunk_size=200)
    chunks = chunker.split(_LONG_TEXT)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "id чанков должны быть уникальными"


def test_fixed_chunker_chunk_index_sequential() -> None:
    chunker = FixedSizeChunker(chunk_size=200)
    chunks = chunker.split(_LONG_TEXT)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_fixed_chunker_empty_text_returns_empty() -> None:
    chunks = FixedSizeChunker().split("   ")
    assert chunks == []


# ---------------------------------------------------------------------------
# ByHeaderChunker
# ---------------------------------------------------------------------------

_MD_TEXT = """# Введение

Это вводный раздел. Здесь объясняется тема.

## Основная часть

Здесь основное содержимое документа.
Может быть несколько абзацев.

## Заключение

Финальные мысли.
"""


def test_by_header_splits_on_headers() -> None:
    chunker = ByHeaderChunker(chunk_size=1000)
    chunks = chunker.split(_MD_TEXT)
    assert len(chunks) == 3
    headers = [c.metadata["header"] for c in chunks]
    assert any("Введение" in h for h in headers)
    assert any("Основная" in h for h in headers)
    assert any("Заключение" in h for h in headers)


def test_by_header_strategy_in_metadata() -> None:
    chunks = ByHeaderChunker().split(_MD_TEXT)
    for chunk in chunks:
        assert chunk.metadata["strategy"] == "by_header"


def test_by_header_no_headers_returns_single_chunk() -> None:
    text = "Просто текст без заголовков. Обычный абзац."
    chunks = ByHeaderChunker(chunk_size=500).split(text)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_by_header_preamble_before_first_header() -> None:
    text = "Вступление перед заголовком.\n\n# Раздел\n\nТекст раздела."
    chunks = ByHeaderChunker(chunk_size=500, min_chunk_size=10).split(text)
    texts = [c.text for c in chunks]
    assert any("Вступление" in t for t in texts)
    assert any("Раздел" in t for t in texts)


def test_by_header_long_section_subdivided() -> None:
    # Раздел длиннее chunk_size → должен быть разбит дополнительно
    long_body = " ".join(["текст"] * 200)
    text = f"# Большой раздел\n\n{long_body}"
    chunks = ByHeaderChunker(chunk_size=200).split(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["header"] == "# Большой раздел"


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

_PARA_TEXT = "\n\n".join([
    "Первый параграф о Python. Он про программирование.",
    "Второй параграф тоже о Python. Динамическая типизация.",
    "Третий параграф о машинном обучении. Совсем другая тема.",
    "Четвёртый параграф тоже о ML. Нейронные сети.",
])


def test_semantic_chunker_size_mode_creates_chunks() -> None:
    chunker = SemanticChunker(chunk_size=100)
    chunks = chunker.split(_PARA_TEXT)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata["strategy"] == "semantic"


def test_semantic_chunker_respects_chunk_size() -> None:
    chunker = SemanticChunker(chunk_size=120)
    chunks = chunker.split(_PARA_TEXT)
    for chunk in chunks:
        # Допустимо небольшое превышение, если абзац сам по себе длиннее лимита
        assert len(chunk.text) <= 120 + 100


def test_semantic_chunker_metadata_inherited() -> None:
    chunks = SemanticChunker(chunk_size=300).split(_PARA_TEXT, metadata={"doc": "x"})
    for chunk in chunks:
        assert chunk.metadata["doc"] == "x"


def test_semantic_chunker_with_embed_fn() -> None:
    import math

    def fake_embed(text: str) -> list[float]:
        # Все параграфы получают одинаковый вектор → высокое сходство → объединятся
        return [1.0, 0.0]

    chunker = SemanticChunker(chunk_size=10_000, similarity_threshold=0.5, embed_fn=fake_embed)
    chunks = chunker.split(_PARA_TEXT)
    # Все параграфы должны слиться в один чанк (высокое сходство + большой лимит)
    assert len(chunks) == 1


def test_semantic_chunker_empty_text() -> None:
    chunks = SemanticChunker().split("   \n\n   ")
    assert chunks == []


def test_semantic_chunker_ids_unique() -> None:
    chunker = SemanticChunker(chunk_size=80)
    chunks = chunker.split(_PARA_TEXT)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# TextLoader
# ---------------------------------------------------------------------------

def test_text_loader(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("Привет, мир!", encoding="utf-8")
    assert TextLoader().load(f) == "Привет, мир!"


def test_text_loader_custom_encoding(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_bytes("Текст".encode("windows-1251"))
    text = TextLoader(encoding="windows-1251").load(f)
    assert "Текст" in text


# ---------------------------------------------------------------------------
# MarkdownLoader
# ---------------------------------------------------------------------------

_MD_CONTENT = "# Заголовок\n\n**Жирный** текст и `код`.\n\n[Ссылка](http://example.com)"


def test_markdown_loader_preserves_markup(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(_MD_CONTENT, encoding="utf-8")
    text = MarkdownLoader(strip_markup=False).load(f)
    assert "# Заголовок" in text
    assert "**Жирный**" in text


def test_markdown_loader_strips_markup(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(_MD_CONTENT, encoding="utf-8")
    text = MarkdownLoader(strip_markup=True).load(f)
    assert "#" not in text
    assert "**" not in text
    assert "`" not in text
    assert "Заголовок" in text
    assert "Жирный" in text


# ---------------------------------------------------------------------------
# HTMLLoader
# ---------------------------------------------------------------------------

_HTML_CONTENT = """<!DOCTYPE html>
<html>
<head><title>Test</title><style>body{color:red}</style></head>
<body>
  <h1>Заголовок страницы</h1>
  <p>Абзац с <strong>жирным</strong> текстом.</p>
  <script>alert('xss')</script>
</body>
</html>"""


def test_html_loader(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text(_HTML_CONTENT, encoding="utf-8")
    text = HTMLLoader().load(f)
    assert "Заголовок страницы" in text
    assert "Абзац" in text
    # скрипты и стили должны быть удалены
    assert "alert" not in text
    assert "color:red" not in text


def test_html_loader_strips_tags(tmp_path: Path) -> None:
    f = tmp_path / "simple.html"
    f.write_text("<p>Текст</p><p>Ещё текст</p>", encoding="utf-8")
    text = HTMLLoader().load(f)
    assert "Текст" in text
    assert "<p>" not in text


# ---------------------------------------------------------------------------
# PDFLoader
# ---------------------------------------------------------------------------

def test_pdf_loader_blank_page(tmp_path: Path) -> None:
    """PDFLoader корректно работает с PDF-файлом (пустая страница → пустая строка)."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_path = tmp_path / "blank.pdf"
    pdf_path.write_bytes(buf.getvalue())

    text = PDFLoader().load(pdf_path)
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------

def test_ingest_txt_fixed(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text(_LONG_TEXT, encoding="utf-8")
    chunks = ingest(f, strategy="fixed", chunk_size=200)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.metadata["filename"] == "doc.txt" for c in chunks)


def test_ingest_md_by_header(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(_MD_TEXT, encoding="utf-8")
    chunks = ingest(f, strategy="by_header", chunk_size=1000)
    assert len(chunks) == 3


def test_ingest_html_fixed(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text(_HTML_CONTENT, encoding="utf-8")
    chunks = ingest(f, strategy="fixed", chunk_size=500)
    assert len(chunks) >= 1
    assert all(c.metadata["filename"] == "page.html" for c in chunks)


def test_ingest_adds_source_metadata(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("Текст документа.", encoding="utf-8")
    chunks = ingest(f, metadata={"author": "Test"})
    assert chunks[0].metadata["author"] == "Test"
    assert "source" in chunks[0].metadata


def test_ingest_unsupported_extension_raises(tmp_path: Path) -> None:
    f = tmp_path / "doc.xyz"
    f.write_text("данные")
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        ingest(f)


def test_ingest_unknown_strategy_raises(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("текст")
    with pytest.raises(ValueError, match="Неизвестная стратегия"):
        ingest(f, strategy="magic")


def test_ingest_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ingest("/nonexistent/path/doc.txt")
