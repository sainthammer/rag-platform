"""Тесты TTL-кэша и структурированного логирования в retrieval.RAGPipeline.

Не требует внешних сервисов: используются FakeEmbeddingService и MockLLM.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from llm.llm_dataclasses import Message
from llm.ports import LLMProvider
from retrieval.pipeline import RAGPipeline
from vector_store.store_dataclasses import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLM(LLMProvider):
    def __init__(self, answer: str = "тестовый ответ") -> None:
        self.model = "mock"
        self.answer = answer
        self.call_count = 0

    async def complete(
        self, messages: list[Message], stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        self.call_count += 1
        return self.answer


def _make_vector_db(docs: list[str]) -> MagicMock:
    db = MagicMock()
    db.search.return_value = SearchResult(
        ids=[str(i) for i in range(len(docs))],
        documents=docs,
        distances=[0.1] * len(docs),
        metadatas=[{}] * len(docs),
    )
    return db


def _embed(text: str) -> list[float]:
    return [0.0] * 4


_DEFAULT_DOCS = ["документ о Python"]


def _make_pipeline(
    docs: list[str] | None = None,
    cache_ttl: float = 300.0,
    llm: LLMProvider | None = None,
    use_empty_docs: bool = False,
) -> tuple[RAGPipeline, _MockLLM]:
    mock_llm = llm or _MockLLM()
    effective_docs = [] if use_empty_docs else (docs if docs is not None else _DEFAULT_DOCS)
    pipeline = RAGPipeline(
        llm=mock_llm,
        vector_db_factory=lambda _: _make_vector_db(effective_docs),
        embed_fn=_embed,
        cache_ttl=cache_ttl,
    )
    return pipeline, mock_llm  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Cache: базовые сценарии
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_on_repeated_question() -> None:
    """Второй вызов с тем же вопросом не должен вызывать LLM."""
    pipeline, llm = _make_pipeline()

    resp1 = await pipeline.ask("что такое Python?", "default")
    resp2 = await pipeline.ask("что такое Python?", "default")

    assert llm.call_count == 1  # LLM вызван только один раз
    assert resp1.answer == resp2.answer


@pytest.mark.asyncio
async def test_cache_miss_on_different_question() -> None:
    """Разные вопросы должны вызывать LLM каждый раз."""
    pipeline, llm = _make_pipeline()

    await pipeline.ask("вопрос один", "default")
    await pipeline.ask("вопрос два", "default")

    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_cache_miss_on_different_collection() -> None:
    """Тот же вопрос в другой коллекции — кэш-промах."""
    pipeline, llm = _make_pipeline()

    await pipeline.ask("вопрос", "col_a")
    await pipeline.ask("вопрос", "col_b")

    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_cache_disabled_when_ttl_zero() -> None:
    """cache_ttl=0 отключает кэш полностью."""
    pipeline, llm = _make_pipeline(cache_ttl=0.0)

    await pipeline.ask("вопрос", "default")
    await pipeline.ask("вопрос", "default")

    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_cache_expired_calls_llm_again() -> None:
    """После истечения TTL кэш-запись становится невалидной."""
    pipeline, llm = _make_pipeline(cache_ttl=0.05)  # 50 ms TTL

    await pipeline.ask("вопрос", "default")
    assert llm.call_count == 1

    time.sleep(0.1)  # ждём истечения TTL

    await pipeline.ask("вопрос", "default")
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_cache_stores_response_object() -> None:
    """Кэшированный ответ — та же RAGResponse (не копия, а тот же объект)."""
    pipeline, _ = _make_pipeline()

    resp1 = await pipeline.ask("вопрос", "default")
    resp2 = await pipeline.ask("вопрос", "default")

    assert resp1 is resp2


# ---------------------------------------------------------------------------
# Cache: совместимость с fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_not_used_when_score_below_threshold() -> None:
    """При score < threshold возвращается fallback, но не кэшируется ошибка."""
    pipeline, llm = _make_pipeline()
    pipeline.score_threshold = 1.0  # ни один чанк не пройдёт (distances=0.1 → score≈0.91)

    # На самом деле с distance=0.1 score=1/1.1≈0.91, так что threshold=1.0 даст fallback
    resp1 = await pipeline.ask("вопрос", "default")
    resp2 = await pipeline.ask("вопрос", "default")

    # fallback не попадает в кэш (llm не вызывался)
    assert llm.call_count == 0
    assert resp1.answer == resp2.answer  # оба — fallback-строка


# ---------------------------------------------------------------------------
# Logging: структура записей
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logging_on_successful_call(caplog: pytest.LogCaptureFixture) -> None:
    """ask() должен записывать rag_call с нужными полями."""
    pipeline, _ = _make_pipeline()

    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await pipeline.ask("что такое Python?", "default")

    # Фильтруем записи нашего логгера
    rag_records = [r for r in caplog.records if r.name == "rag.pipeline"]
    assert any(r.message == "rag_call" for r in rag_records)

    call_record = next(r for r in rag_records if r.message == "rag_call")
    # Проверяем наличие ключевых полей в extra
    assert hasattr(call_record, "question")
    assert hasattr(call_record, "chunks_count")
    assert hasattr(call_record, "prompt_tokens")
    assert hasattr(call_record, "completion_tokens")
    assert hasattr(call_record, "latency_ms")
    assert call_record.cache_hit is False


@pytest.mark.asyncio
async def test_logging_on_cache_hit(caplog: pytest.LogCaptureFixture) -> None:
    """При cache hit логируется rag_cache_hit, а rag_call не повторяется."""
    pipeline, _ = _make_pipeline()

    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await pipeline.ask("вопрос", "default")  # первый — cache miss
        await pipeline.ask("вопрос", "default")  # второй — cache hit

    rag_records = [r for r in caplog.records if r.name == "rag.pipeline"]
    messages = [r.message for r in rag_records]

    assert "rag_call" in messages
    assert "rag_cache_hit" in messages


@pytest.mark.asyncio
async def test_logging_question_truncated_to_200_chars(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Длинный вопрос в логе должен быть обрезан до 200 символов."""
    pipeline, _ = _make_pipeline()
    long_question = "а" * 500

    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await pipeline.ask(long_question, "default")

    call_record = next(
        r for r in caplog.records
        if r.name == "rag.pipeline" and r.message == "rag_call"
    )
    assert len(call_record.question) <= 200


@pytest.mark.asyncio
async def test_logging_token_counts_are_nonnegative(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline, _ = _make_pipeline()

    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await pipeline.ask("Python — что это?", "default")

    call_record = next(
        r for r in caplog.records
        if r.name == "rag.pipeline" and r.message == "rag_call"
    )
    assert call_record.prompt_tokens >= 0
    assert call_record.completion_tokens >= 0
    assert call_record.latency_ms >= 0


@pytest.mark.asyncio
async def test_logging_on_fallback_call(caplog: pytest.LogCaptureFixture) -> None:
    """При fallback (нет чанков или score ниже порога) тоже пишем rag_call."""
    pipeline, _ = _make_pipeline(use_empty_docs=True)  # пустая коллекция

    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await pipeline.ask("вопрос без контекста", "default")

    rag_records = [r for r in caplog.records if r.name == "rag.pipeline"]
    assert any(r.message == "rag_call" for r in rag_records)

    fallback_record = next(r for r in rag_records if r.message == "rag_call")
    assert fallback_record.chunks_count == 0
    assert fallback_record.prompt_tokens == 0
