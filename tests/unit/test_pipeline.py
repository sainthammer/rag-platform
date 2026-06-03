from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from llm.llm_dataclasses import Message
from llm.pipeline import RAGPipeline
from llm.ports import LLMProvider
from llm.prompt_templates import BASE, CITATION, STRICT, get_template
from llm.token_budget import TokenBudgetManager
from vector_store.store_dataclasses import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    """Captures every complete() call for later inspection."""

    def __init__(self, response: str = "mocked answer") -> None:
        self.response = response
        self.calls: list[tuple[list[Message], bool]] = []

    async def complete(
        self, messages: list[Message], stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        self.calls.append((messages, stream))
        return self.response


def _make_vector_db(documents: list[str]) -> MagicMock:
    db = MagicMock()
    db.search.return_value = SearchResult(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
        distances=[0.1] * len(documents),
        metadatas=[{}] * len(documents),
    )
    return db


def _embed(text: str) -> list[float]:
    return [0.0] * 4


# ---------------------------------------------------------------------------
# RAGPipeline — message structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_sends_system_and_user_messages() -> None:
    provider = MockProvider()
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db(["Paris is the capital of France."]),
        embed_fn=_embed,
    )

    await pipeline.run("What is the capital of France?")

    assert len(provider.calls) == 1
    messages, stream = provider.calls[0]
    assert stream is False
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"


@pytest.mark.asyncio
async def test_pipeline_injects_context_into_system_message() -> None:
    doc = "Berlin is the capital of Germany."
    provider = MockProvider()
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db([doc]),
        embed_fn=_embed,
    )

    await pipeline.run("Capital of Germany?")

    system_content = provider.calls[0][0][0].content
    assert doc in system_content


@pytest.mark.asyncio
async def test_pipeline_passes_query_as_user_message() -> None:
    provider = MockProvider()
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db(["irrelevant doc"]),
        embed_fn=_embed,
    )
    query = "What is photosynthesis?"

    await pipeline.run(query)

    user_content = provider.calls[0][0][1].content
    assert query in user_content


@pytest.mark.asyncio
async def test_pipeline_returns_provider_response() -> None:
    provider = MockProvider(response="42")
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db([]),
        embed_fn=_embed,
    )

    result = await pipeline.run("answer?")
    assert result == "42"


@pytest.mark.asyncio
async def test_pipeline_passes_stream_flag() -> None:
    provider = MockProvider()
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db(["doc"]),
        embed_fn=_embed,
    )

    await pipeline.run("q", stream=True)

    _, stream = provider.calls[0]
    assert stream is True


# ---------------------------------------------------------------------------
# RAGPipeline — embed_fn is called with the query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_calls_embed_fn_with_query() -> None:
    captured: list[str] = []

    def embed(text: str) -> list[float]:
        captured.append(text)
        return [0.0] * 4

    provider = MockProvider()
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db([]),
        embed_fn=embed,
    )
    await pipeline.run("my query")

    assert captured == ["my query"]


# ---------------------------------------------------------------------------
# RAGPipeline — prompt templates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_strict_template_adds_user_prefix() -> None:
    provider = MockProvider()
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db(["ctx"]),
        embed_fn=_embed,
        template=STRICT,
    )

    await pipeline.run("question")

    user_content = provider.calls[0][0][1].content
    assert "придумывай" in user_content.lower()


@pytest.mark.asyncio
async def test_pipeline_citation_template_system_mentions_citations() -> None:
    provider = MockProvider()
    pipeline = RAGPipeline(
        llm=provider,
        vector_db=_make_vector_db(["ctx"]),
        embed_fn=_embed,
        template=CITATION,
    )

    await pipeline.run("question")

    system_content = provider.calls[0][0][0].content
    assert "источник" in system_content.lower()


# ---------------------------------------------------------------------------
# TokenBudgetManager — chunk fitting
# ---------------------------------------------------------------------------


def test_budget_fits_all_chunks_within_budget() -> None:
    budget = TokenBudgetManager(model="gpt-4o", max_context_tokens=500, reserved_tokens=0)
    short_chunks = ["hello world"] * 5
    result = budget.fit_chunks(short_chunks)
    assert result == short_chunks


def test_budget_truncates_chunks_exceeding_limit() -> None:
    budget = TokenBudgetManager(model="gpt-4o", max_context_tokens=10, reserved_tokens=0)
    # Each word is ~1 token; 20 words will exceed budget of 10
    chunks = ["word " * 10, "word " * 10]
    result = budget.fit_chunks(chunks)
    assert len(result) < len(chunks)


def test_budget_empty_input_returns_empty() -> None:
    budget = TokenBudgetManager()
    assert budget.fit_chunks([]) == []


def test_budget_count_returns_positive_for_nonempty_text() -> None:
    budget = TokenBudgetManager()
    assert budget.count("hello world") > 0


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


def test_get_template_returns_correct_template() -> None:
    assert get_template("base") is BASE
    assert get_template("strict") is STRICT
    assert get_template("citation") is CITATION


def test_get_template_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown template"):
        get_template("nonexistent")


def test_base_template_format_system_contains_context() -> None:
    ctx = "some context text"
    rendered = BASE.format_system(ctx)
    assert ctx in rendered


def test_base_template_format_user_returns_query_unchanged() -> None:
    q = "what is life?"
    assert BASE.format_user(q) == q


def test_strict_template_format_user_prepends_prefix() -> None:
    q = "my question"
    result = STRICT.format_user(q)
    assert result.endswith(q)
    assert STRICT.user_prefix in result
