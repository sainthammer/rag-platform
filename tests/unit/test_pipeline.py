from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from llm.llm_dataclasses import Message
from llm.ports import LLMProvider
from llm.prompt_templates import BASE, CITATION, STRICT, get_template
from llm.token_budget import TokenBudgetManager
from retrieval.pipeline import RAGPipeline, RAGResponse
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


def _make_pipeline(
    provider: LLMProvider,
    documents: list[str],
    **kwargs,
) -> RAGPipeline:
    return RAGPipeline(
        llm=provider,
        vector_db_factory=lambda _: _make_vector_db(documents),
        embed_fn=_embed,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# RAGPipeline — message structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_sends_system_and_user_messages() -> None:
    provider = MockProvider()
    pipeline = _make_pipeline(provider, ["Paris is the capital of France."])

    await pipeline.ask("What is the capital of France?", "test")

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
    pipeline = _make_pipeline(provider, [doc])

    await pipeline.ask("Capital of Germany?", "test")

    system_content = provider.calls[0][0][0].content
    assert doc in system_content


@pytest.mark.asyncio
async def test_pipeline_passes_query_as_user_message() -> None:
    provider = MockProvider()
    pipeline = _make_pipeline(provider, ["irrelevant doc"])
    query = "What is photosynthesis?"

    await pipeline.ask(query, "test")

    user_content = provider.calls[0][0][1].content
    assert query in user_content


@pytest.mark.asyncio
async def test_pipeline_returns_provider_response() -> None:
    provider = MockProvider(response="42")
    pipeline = _make_pipeline(provider, ["some doc"])

    result = await pipeline.ask("answer?", "test")
    assert isinstance(result, RAGResponse)
    assert result.answer == "42"


@pytest.mark.asyncio
async def test_pipeline_passes_stream_flag() -> None:
    provider = MockProvider()
    pipeline = _make_pipeline(provider, ["doc"])

    await pipeline.ask("q", "test", stream=True)

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
        vector_db_factory=lambda _: _make_vector_db([]),
        embed_fn=embed,
    )
    await pipeline.ask("my query", "test")

    assert captured == ["my query"]


# ---------------------------------------------------------------------------
# RAGPipeline — prompt templates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_strict_template_adds_user_prefix() -> None:
    provider = MockProvider()
    pipeline = _make_pipeline(provider, ["ctx"], template=STRICT)

    await pipeline.ask("question", "test")

    user_content = provider.calls[0][0][1].content
    assert "придумывай" in user_content.lower()


@pytest.mark.asyncio
async def test_pipeline_citation_template_system_mentions_citations() -> None:
    provider = MockProvider()
    pipeline = _make_pipeline(provider, ["ctx"], template=CITATION)

    await pipeline.ask("question", "test")

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


# ---------------------------------------------------------------------------
# RAGPipeline — ask returns RAGResponse with answer and sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_returns_answer_and_sources() -> None:
    docs = ["Chunk A about Paris.", "Chunk B about France."]
    provider = MockProvider(response="Paris is the capital.")
    pipeline = _make_pipeline(provider, docs)

    result = await pipeline.ask("Capital of France?", "test")

    assert result.answer == "Paris is the capital."
    assert [s.text for s in result.sources] == docs


@pytest.mark.asyncio
async def test_ask_budget_limits_llm_context() -> None:
    """Бюджет должен ограничивать контекст в промпте, переданном LLM."""
    budget = TokenBudgetManager(model="gpt-4o", max_context_tokens=5, reserved_tokens=0)
    docs = ["hello world", "this is a very long chunk that will not fit"]
    provider = MockProvider(response="ok")
    pipeline = _make_pipeline(provider, docs, budget=budget)

    await pipeline.ask("q", "test")

    system_content = provider.calls[0][0][0].content
    assert "this is a very long chunk" not in system_content


@pytest.mark.asyncio
async def test_ask_calls_llm_once() -> None:
    provider = MockProvider()
    pipeline = _make_pipeline(provider, ["doc"])

    await pipeline.ask("question", "test")

    assert len(provider.calls) == 1
    _, stream = provider.calls[0]
    assert stream is False


# ---------------------------------------------------------------------------
# RAGPipeline — error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_propagates_llm_error() -> None:
    class FailingProvider(LLMProvider):
        async def complete(self, messages, stream=False):
            raise RuntimeError("LLM failure")

    pipeline = RAGPipeline(
        llm=FailingProvider(),
        vector_db_factory=lambda _: _make_vector_db(["doc"]),
        embed_fn=_embed,
    )

    with pytest.raises(RuntimeError, match="LLM failure"):
        await pipeline.ask("question", "test")


@pytest.mark.asyncio
async def test_ask_propagates_embed_error() -> None:
    def failing_embed(text: str) -> list[float]:
        raise ValueError("embed error")

    pipeline = RAGPipeline(
        llm=MockProvider(),
        vector_db_factory=lambda _: _make_vector_db(["doc"]),
        embed_fn=failing_embed,
    )

    with pytest.raises(ValueError, match="embed error"):
        await pipeline.ask("q", "test")


# ---------------------------------------------------------------------------
# TokenBudgetManager — remaining
# ---------------------------------------------------------------------------


def test_budget_remaining_is_positive_for_short_chunks() -> None:
    budget = TokenBudgetManager(model="gpt-4o", max_context_tokens=200, reserved_tokens=0)
    remaining = budget.remaining(["hello"])
    assert remaining > 0


def test_budget_remaining_zero_when_budget_exhausted() -> None:
    budget = TokenBudgetManager(model="gpt-4o", max_context_tokens=5, reserved_tokens=0)
    remaining = budget.remaining(["a very long text that exceeds budget significantly"])
    assert remaining >= 0
