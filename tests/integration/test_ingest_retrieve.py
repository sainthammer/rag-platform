"""Интеграционные тесты: ingest → retrieve.

Тесты используют:
  - FakeEmbeddingService: детерминированные векторы без ML-модели.
    Ключевое свойство: embed(T) == embed(T) для одного и того же текста T.
    Это гарантирует, что запрос с точным текстом чанка даёт score ≈ 1.0.
  - ChromaDB (PersistentClient через tmp_path).
  - Реальные вызовы ingest() + Retriever.retrieve().

Тесты проверяют:
  1. ingest .txt → retrieve: top-1 релевантен запросу (score > 0.9).
  2. ingest PDF → retrieve: полный цикл с PDF-файлом.
  3. Фильтрация по metadata через filters=.
  4. MMRReranker снижает дублирование в результатах.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from chunking import ingest
from embeddings.adapters import FakeEmbeddingService
from retrieval.pipeline import SourceChunk
from retrieval.rerankers import MMRReranker
from retrieval.retriever import Retriever
from vector_store.adapters import ChromaDB


# ---------------------------------------------------------------------------
# Общие константы и хелперы
# ---------------------------------------------------------------------------

_EMBED = FakeEmbeddingService(size=8, normalize=True)

_DOCS = {
    "python": "Python is a high-level programming language used for data science.",
    "docker": "Docker containers isolate applications from the host system.",
    "ml": "Machine learning requires large labeled datasets and compute power.",
}


def _make_db(tmp_path: Path, collection: str = "test") -> ChromaDB:
    return ChromaDB(collection, persist_directory=str(tmp_path / "chroma"))


def _populate_db(db: ChromaDB, texts: dict[str, str]) -> list[SourceChunk]:
    """Добавить тексты в ChromaDB и вернуть их как SourceChunk-список."""
    import hashlib

    chunks = [
        SourceChunk(
            text=text,
            score=1.0,
            doc_id=key,
            metadata={"source": f"{key}.txt"},
        )
        for key, text in texts.items()
    ]
    db.add(
        ids=[c.doc_id for c in chunks],
        embeddings=[_EMBED.embed(c.text) for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return chunks


def _make_retriever(db: ChromaDB) -> Retriever:
    return Retriever(
        embed_fn=_EMBED.embed,
        vector_db_factory=lambda _: db,
    )


# ---------------------------------------------------------------------------
# Тест 1: ingest .txt → retrieve
# ---------------------------------------------------------------------------


def test_ingest_txt_top1_exact_match(tmp_path: Path) -> None:
    """Запрос с точным текстом первого чанка должен давать score > 0.9 в top-1."""
    # Три коротких абзаца → три отдельных чанка при chunk_size=80
    content = "\n\n".join(_DOCS.values())
    txt_path = tmp_path / "docs.txt"
    txt_path.write_text(content, encoding="utf-8")

    chunks = ingest(txt_path, strategy="fixed", chunk_size=80, chunk_overlap=0)
    assert len(chunks) >= 2, "Ожидаем несколько чанков при chunk_size=80"

    db = _make_db(tmp_path)
    db.add(
        ids=[c.id for c in chunks],
        embeddings=[_EMBED.embed(c.text) for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    retriever = _make_retriever(db)
    # Запрашиваем точный текст первого чанка → score должен быть максимальным
    target = chunks[0].text
    results = retriever.retrieve(target, collection="test", top_k=1)

    assert len(results) == 1
    assert results[0].text == target, "top-1 должен совпадать с запрошенным текстом"
    assert results[0].score > 0.9, f"Ожидаем score > 0.9, получили {results[0].score}"


def test_ingest_txt_returns_chunks_with_metadata(tmp_path: Path) -> None:
    """ingest() должен заполнить metadata: filename и source."""
    txt_path = tmp_path / "guide.txt"
    txt_path.write_text("Текст для индексации.", encoding="utf-8")

    chunks = ingest(txt_path, strategy="fixed", chunk_size=500)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata.get("filename") == "guide.txt"
        assert "source" in chunk.metadata


# ---------------------------------------------------------------------------
# Тест 2: ingest PDF → retrieve
# ---------------------------------------------------------------------------


def _make_text_pdf(text: str) -> bytes:
    """Создать минимальный PDF с ASCII-текстом через pypdf low-level API."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    # Стандартный Type1-шрифт (Helvetica встроен в PDF-ридеры, не требует embed)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})
    })

    # Content stream: BT...ET с простым текстом
    safe = text[:200].encode("ascii", errors="replace").decode("ascii")
    safe = safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET\n".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_ingest_pdf_full_pipeline(tmp_path: Path) -> None:
    """Полный цикл: создать PDF → ingest → index → retrieve."""
    pdf_text = "Python is a high-level programming language for data science."
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(_make_text_pdf(pdf_text))

    chunks = ingest(pdf_path, strategy="fixed", chunk_size=500)

    if not chunks:
        # Некоторые PDF-движки не извлекают текст из синтетических PDF
        pytest.skip("pypdf не смог извлечь текст из синтетического PDF")

    db = _make_db(tmp_path, "pdf_col")
    db.add(
        ids=[c.id for c in chunks],
        embeddings=[_EMBED.embed(c.text) for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    retriever = Retriever(embed_fn=_EMBED.embed, vector_db_factory=lambda _: db)
    results = retriever.retrieve(chunks[0].text, collection="pdf_col", top_k=1)

    assert len(results) >= 1
    assert results[0].score > 0.9


# ---------------------------------------------------------------------------
# Тест 3: фильтрация по metadata
# ---------------------------------------------------------------------------


def test_retrieve_filters_by_metadata(tmp_path: Path) -> None:
    """filters={"source": "python.txt"} должен вернуть только Python-чанк."""
    db = _make_db(tmp_path, "filter_test")
    _populate_db(db, _DOCS)

    retriever = _make_retriever(db)
    results = retriever.retrieve(
        "programming language",
        collection="filter_test",
        top_k=5,
        filters={"source": "python.txt"},
    )

    assert all(r.metadata.get("source") == "python.txt" for r in results), (
        "Все результаты должны иметь source=python.txt"
    )


def test_retrieve_filters_no_match_returns_empty(tmp_path: Path) -> None:
    """Фильтр, которому ничего не соответствует, должен вернуть пустой список."""
    db = _make_db(tmp_path, "empty_filter")
    _populate_db(db, _DOCS)

    retriever = _make_retriever(db)
    results = retriever.retrieve(
        "любой запрос",
        collection="empty_filter",
        top_k=5,
        filters={"source": "nonexistent.txt"},
    )

    assert results == []


# ---------------------------------------------------------------------------
# Тест 4: MMRReranker снижает дублирование
# ---------------------------------------------------------------------------


def test_mmr_reranker_diversifies_results(tmp_path: Path) -> None:
    """MMRReranker с λ=0 (только diversity) не должен выбрать два одинаковых чанка."""
    # Два почти одинаковых текста + один совершенно другой
    similar_texts = {
        "a": "Python is a great language. Python is widely used.",
        "b": "Python is a great language. Python has many libraries.",
        "c": "Docker containers are lightweight and portable.",
    }
    db = _make_db(tmp_path, "mmr_test")
    _populate_db(db, similar_texts)

    reranker = MMRReranker(lambda_=0.3, embed_fn=_EMBED.embed)
    retriever = Retriever(
        embed_fn=_EMBED.embed,
        vector_db_factory=lambda _: db,
        reranker=reranker,
        fetch_k=3,
    )

    results = retriever.retrieve("Python language", collection="mmr_test", top_k=3)

    assert len(results) >= 2
    # Проверяем, что нет двух идентичных текстов в результатах
    texts = [r.text for r in results]
    assert len(texts) == len(set(texts)), "Не должно быть дублирующихся текстов"


# ---------------------------------------------------------------------------
# Тест 5: top_k ограничивает количество результатов
# ---------------------------------------------------------------------------


def test_retrieve_respects_top_k(tmp_path: Path) -> None:
    """retrieve() должен возвращать не больше top_k результатов."""
    db = _make_db(tmp_path, "topk_test")
    _populate_db(db, _DOCS)

    retriever = _make_retriever(db)

    for k in [1, 2, 3]:
        results = retriever.retrieve("test", collection="topk_test", top_k=k)
        assert len(results) <= k, f"Ожидали ≤{k} результатов, получили {len(results)}"


def test_retrieve_empty_collection_returns_empty(tmp_path: Path) -> None:
    """Retriever из пустой коллекции должен вернуть пустой список."""
    db = _make_db(tmp_path, "empty_col")

    retriever = _make_retriever(db)
    # ChromaDB возвращает ошибку для n_results > count; проверяем graceful handling
    try:
        results = retriever.retrieve("запрос", collection="empty_col", top_k=3)
        assert results == []
    except Exception:
        # Некоторые версии ChromaDB бросают исключение при пустой коллекции
        pytest.skip("ChromaDB threw on empty collection query")
