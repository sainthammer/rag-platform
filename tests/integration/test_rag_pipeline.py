"""Интеграционный тест RAGPipeline: реальный ChromaDB + mock LLM.

Тесты используют:
  - реальный ChromaDB (PersistentClient через tmp_path pytest);
  - детерминированный fake_embed() — SHA256-хэш без внешних зависимостей;
  - MockLLM — захватывает вызовы, возвращает предсказуемый ответ.

Проверяется:
  - ask() возвращает sources с текстами из ChromaDB;
  - sources содержат корректные metadata и doc_id;
  - fallback при пустой коллекции (нет чанков);
  - fallback при score_threshold выше любого возможного score;
  - ask_stream() отдаёт токены и применяет fallback;
  - ask_multi() запрашивает несколько коллекций параллельно.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import AsyncGenerator

import pytest

from llm.llm_dataclasses import Message
from llm.ports import LLMProvider
from retrieval.pipeline import FALLBACK_ANSWER, RAGPipeline, RAGResponse, SourceChunk
from vector_store.adapters import ChromaDB

# ---------------------------------------------------------------------------
# Вспомогательный fake-embedding (без внешних зависимостей)
# ---------------------------------------------------------------------------

_EMBED_DIM = 8


def fake_embed(text: str) -> list[float]:
    """Детерминированный unit-norm вектор на базе SHA256."""
    digest = hashlib.sha256(text.encode()).digest()
    values = [digest[i % len(digest)] / 255.0 for i in range(_EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm > 0 else values


# ---------------------------------------------------------------------------
# Тестовые данные
# ---------------------------------------------------------------------------

_DOCS = [
    "RAG объединяет поиск по документам и генерацию ответа.",
    "Vector store хранит embedding-векторы чанков документов.",
    "ChromaDB — векторная база данных с поддержкой персистентности.",
]
_METADATAS = [{"source": f"file{i}.txt"} for i in range(len(_DOCS))]


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------


class MockLLM(LLMProvider):
    """Захватывает все вызовы complete() и возвращает предсказуемый ответ."""

    def __init__(self, answer: str = "Ответ от mock LLM") -> None:
        self.answer = answer
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        self.calls.append(messages)
        if stream:
            answer = self.answer

            async def _gen() -> AsyncGenerator[str, None]:
                for token in answer.split():
                    yield token + " "

            return _gen()
        return self.answer


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture()
def populated_db(tmp_path: Path) -> ChromaDB:
    """ChromaDB с тремя проиндексированными документами."""
    db = ChromaDB(collection="docs", persist_directory=str(tmp_path / "chroma"))
    db.add(
        ids=[f"doc{i}" for i in range(len(_DOCS))],
        embeddings=[fake_embed(doc) for doc in _DOCS],
        documents=_DOCS,
        metadatas=_METADATAS,
    )
    return db


@pytest.fixture()
def empty_db(tmp_path: Path) -> ChromaDB:
    """ChromaDB без документов."""
    return ChromaDB(collection="empty", persist_directory=str(tmp_path / "empty"))


def _make_pipeline(
    db: ChromaDB,
    llm: MockLLM | None = None,
    score_threshold: float = 0.0,
) -> RAGPipeline:
    return RAGPipeline(
        llm=llm or MockLLM(),
        vector_db_factory=lambda _: db,
        embed_fn=fake_embed,
        n_results=3,
        score_threshold=score_threshold,
    )


# ---------------------------------------------------------------------------
# ask() — базовый контракт
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_returns_rag_response(populated_db: ChromaDB) -> None:
    llm = MockLLM()
    pipeline = _make_pipeline(populated_db, llm)

    response = await pipeline.ask("Что такое RAG?", collection="docs")

    assert isinstance(response, RAGResponse)
    assert response.answer == llm.answer
    assert response.latency_ms >= 0
    assert 0.0 <= response.confidence <= 1.0


@pytest.mark.asyncio
async def test_ask_sources_contain_indexed_documents(populated_db: ChromaDB) -> None:
    pipeline = _make_pipeline(populated_db)

    response = await pipeline.ask("Что такое RAG?", collection="docs")

    assert len(response.sources) > 0
    source_texts = {s.text for s in response.sources}
    assert source_texts <= set(_DOCS), "sources должны содержать только проиндексированные тексты"


@pytest.mark.asyncio
async def test_ask_sources_have_correct_metadata(populated_db: ChromaDB) -> None:
    pipeline = _make_pipeline(populated_db)

    response = await pipeline.ask("Vector store", collection="docs")

    for source in response.sources:
        assert isinstance(source, SourceChunk)
        assert "source" in source.metadata
        assert source.metadata["source"].endswith(".txt")
        assert 0.0 < source.score <= 1.0


@pytest.mark.asyncio
async def test_ask_confidence_equals_max_source_score(populated_db: ChromaDB) -> None:
    pipeline = _make_pipeline(populated_db)

    response = await pipeline.ask("ChromaDB", collection="docs")

    max_score = max(s.score for s in response.sources)
    assert response.confidence == pytest.approx(max_score)


@pytest.mark.asyncio
async def test_ask_llm_receives_context_with_retrieved_texts(populated_db: ChromaDB) -> None:
    llm = MockLLM()
    pipeline = _make_pipeline(populated_db, llm)

    await pipeline.ask("RAG", collection="docs")

    assert len(llm.calls) == 1
    system_content = llm.calls[0][0].content
    assert any(doc in system_content for doc in _DOCS), (
        "системный промпт должен содержать хотя бы один проиндексированный документ"
    )


# ---------------------------------------------------------------------------
# ask() — fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_fallback_on_empty_collection(empty_db: ChromaDB) -> None:
    llm = MockLLM()
    pipeline = _make_pipeline(empty_db, llm)

    response = await pipeline.ask("вопрос", collection="empty")

    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []
    assert not llm.calls, "LLM не должен вызываться при отсутствии чанков"


@pytest.mark.asyncio
async def test_ask_fallback_when_score_below_threshold(populated_db: ChromaDB) -> None:
    llm = MockLLM()
    # score ≤ 1.0 всегда, поэтому threshold=1.1 гарантирует fallback
    pipeline = _make_pipeline(populated_db, llm, score_threshold=1.1)

    response = await pipeline.ask("вопрос", collection="docs")

    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []
    assert not llm.calls


# ---------------------------------------------------------------------------
# ask_stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_stream_yields_tokens(populated_db: ChromaDB) -> None:
    llm = MockLLM(answer="один два три")
    pipeline = _make_pipeline(populated_db, llm)

    tokens: list[str] = []
    async for token in pipeline.ask_stream("RAG", collection="docs"):
        tokens.append(token)

    assert len(tokens) > 0
    assert "".join(tokens).strip() == "один два три"


@pytest.mark.asyncio
async def test_ask_stream_fallback_on_empty_collection(empty_db: ChromaDB) -> None:
    llm = MockLLM()
    pipeline = _make_pipeline(empty_db, llm)

    tokens: list[str] = []
    async for token in pipeline.ask_stream("вопрос", collection="empty"):
        tokens.append(token)

    assert "".join(tokens) == FALLBACK_ANSWER
    assert not llm.calls


# ---------------------------------------------------------------------------
# ask_multi() — параллельные коллекции
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_multi_returns_one_response_per_collection(populated_db: ChromaDB) -> None:
    pipeline = _make_pipeline(populated_db)

    responses = await pipeline.ask_multi("RAG", ["col-a", "col-b", "col-c"])

    assert len(responses) == 3
    assert all(isinstance(r, RAGResponse) for r in responses)


@pytest.mark.asyncio
async def test_ask_multi_all_collections_get_answer(populated_db: ChromaDB) -> None:
    llm = MockLLM()
    pipeline = _make_pipeline(populated_db, llm)

    responses = await pipeline.ask_multi("ChromaDB", ["a", "b"])

    assert all(r.answer == llm.answer for r in responses)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_ask_multi_empty_list_returns_empty(populated_db: ChromaDB) -> None:
    pipeline = _make_pipeline(populated_db)

    responses = await pipeline.ask_multi("вопрос", [])

    assert responses == []


# ---------------------------------------------------------------------------
# ask() с stream=True — накопление в RAGResponse.answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_stream_true_accumulates_full_answer(populated_db: ChromaDB) -> None:
    llm = MockLLM(answer="стриминг работает")
    pipeline = _make_pipeline(populated_db, llm)

    response = await pipeline.ask("RAG", collection="docs", stream=True)

    assert isinstance(response, RAGResponse)
    # MockLLM в streaming-режиме добавляет пробел после каждого слова
    assert "стриминг" in response.answer
    assert "работает" in response.answer
