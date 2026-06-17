from pathlib import Path

import pytest

from chunking import (
    ByHeaderChunker,
    Chunk,
    FixedSizeChunker,
    HTMLLoader,
    MarkdownLoader,
    SemanticChunker,
    TextLoader,
    ingest,
)


def test_text_loader_returns_text_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("Hello   world\n", encoding="utf-8")

    text, metadata = TextLoader().load(path)

    assert text == "Hello world"
    assert metadata["source_name"] == "doc.txt"
    assert metadata["content_type"] == "text/plain"
    assert "document_hash" in metadata


def test_markdown_loader_preserves_headers(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nBody", encoding="utf-8")

    text, metadata = MarkdownLoader().load(path)

    assert "# Title" in text
    assert metadata["content_type"] == "text/markdown"


def test_html_loader_extracts_visible_text(tmp_path: Path) -> None:
    pytest.importorskip("bs4")
    path = tmp_path / "doc.html"
    path.write_text(
        "<html><head><style>.x{}</style></head><body><h1>Title</h1><script>x()</script><p>Body</p></body></html>",
        encoding="utf-8",
    )

    text, metadata = HTMLLoader().load(path)

    assert "Title" in text
    assert "Body" in text
    assert "x()" not in text
    assert metadata["content_type"] == "text/html"


def test_fixed_size_chunker_splits_with_metadata() -> None:
    chunks = FixedSizeChunker(chunk_size=5, overlap=1).chunk("abcdefghij", {"source": "doc"})

    assert [chunk.text for chunk in chunks] == ["abcde", "efghi", "ij"]
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].metadata["chunk_strategy"] == "fixed"
    assert chunks[0].id


def test_by_header_chunker_sets_section_metadata() -> None:
    text = "# Intro\nfirst\n\n# Details\nsecond"

    chunks = ByHeaderChunker(chunk_size=50).chunk(text, {"source": "doc.md"})

    assert [chunk.metadata["section"] for chunk in chunks] == ["Intro", "Details"]
    assert chunks[1].text.startswith("# Details")


def test_semantic_chunker_prefers_sentence_boundaries() -> None:
    text = "First sentence. Second sentence. Third sentence."

    chunks = SemanticChunker(chunk_size=28, overlap_sentences=0).chunk(text, {"source": "doc"})

    assert [chunk.text for chunk in chunks] == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    assert chunks[0].metadata["chunk_strategy"] == "semantic"


def test_ingest_selects_loader_and_strategy(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# A\ntext\n\n# B\nmore text", encoding="utf-8")

    chunks = ingest(path, strategy="by_header", chunk_size=100)

    assert len(chunks) == 2
    assert chunks[0].metadata["source_name"] == "doc.md"
    assert chunks[0].metadata["chunk_strategy"] == "by_header"


def test_ingest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "missing.txt")
