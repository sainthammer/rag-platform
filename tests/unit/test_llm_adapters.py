"""Unit-тесты для LLM-адаптеров (OpenAI, Anthropic, Ollama).

Все тесты работают без реальных API-ключей:
- Инициализация провайдера обходится через __new__ + ручная инъекция mock-клиента.
- Async-методы мокаются через AsyncMock и кастомные async-итераторы.
"""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.adapters import AnthropicProvider, OllamaProvider, OpenAIProvider
from llm.llm_dataclasses import Message


# ---------------------------------------------------------------------------
# Вспомогательные async-итераторы для эмуляции streaming
# ---------------------------------------------------------------------------


class _AsyncIter:
    """Обёртка над синхронным итерируемым для async for."""

    def __init__(self, items: list) -> None:
        self._iter = iter(items)

    def __aiter__(self) -> "_AsyncIter":
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _AsyncCM:
    """Async context manager, возвращающий переданный объект."""

    def __init__(self, obj) -> None:
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *_):
        pass


def _openai_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


def _openai_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


def _make_openai(model: str = "gpt-4o-mini") -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = model
    provider.client = MagicMock()
    return provider


@pytest.mark.asyncio
async def test_openai_complete_no_stream_returns_content() -> None:
    provider = _make_openai()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_openai_response("ответ от OpenAI")
    )

    result = await provider.complete([Message(role="user", content="вопрос")], stream=False)

    assert result == "ответ от OpenAI"
    provider.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_complete_no_stream_none_content_returns_empty() -> None:
    provider = _make_openai()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_openai_response(None)  # type: ignore[arg-type]
    )
    # None content → fallback ""
    resp = MagicMock()
    resp.choices[0].message.content = None
    provider.client.chat.completions.create = AsyncMock(return_value=resp)

    result = await provider.complete([Message(role="user", content="вопрос")])
    assert result == ""


@pytest.mark.asyncio
async def test_openai_complete_stream_yields_tokens() -> None:
    provider = _make_openai()

    chunks = [
        _openai_chunk("При"),
        _openai_chunk("вет"),
        _openai_chunk(None),    # служебный чанк с None — должен пропускаться
        _openai_chunk("!"),
    ]
    provider.client.chat.completions.create = AsyncMock(
        return_value=_AsyncIter(chunks)
    )

    gen = await provider.complete([Message(role="user", content="q")], stream=True)
    assert isinstance(gen, AsyncGenerator)

    tokens = [t async for t in gen]
    assert tokens == ["При", "вет", "!"]


@pytest.mark.asyncio
async def test_openai_complete_passes_messages_as_payload() -> None:
    provider = _make_openai()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_openai_response("ok")
    )

    messages = [
        Message(role="system", content="ты ассистент"),
        Message(role="user", content="привет"),
    ]
    await provider.complete(messages)

    call_kwargs = provider.client.chat.completions.create.call_args.kwargs
    payload = call_kwargs.get("messages") or provider.client.chat.completions.create.call_args.args[0] if not call_kwargs.get("messages") else call_kwargs["messages"]
    assert any(m["role"] == "system" for m in payload)
    assert any(m["role"] == "user" for m in payload)


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


def _make_anthropic(model: str = "claude-haiku-4-5-20251001") -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = model
    provider.max_tokens = 1024
    provider.client = MagicMock()
    return provider


def test_anthropic_split_messages_extracts_system() -> None:
    provider = _make_anthropic()
    messages = [
        Message(role="system", content="Ты ассистент."),
        Message(role="user", content="Привет!"),
    ]
    system, turns = provider._split_messages(messages)

    assert system == "Ты ассистент."
    assert len(turns) == 1
    assert turns[0]["role"] == "user"


def test_anthropic_split_messages_no_system() -> None:
    provider = _make_anthropic()
    messages = [Message(role="user", content="вопрос")]
    system, turns = provider._split_messages(messages)

    assert system is None
    assert len(turns) == 1


def test_anthropic_split_messages_multiple_turns() -> None:
    provider = _make_anthropic()
    messages = [
        Message(role="system", content="инструкция"),
        Message(role="user", content="вопрос 1"),
        Message(role="assistant", content="ответ 1"),
        Message(role="user", content="вопрос 2"),
    ]
    system, turns = provider._split_messages(messages)

    assert system == "инструкция"
    assert len(turns) == 3


@pytest.mark.asyncio
async def test_anthropic_complete_no_stream_returns_text() -> None:
    provider = _make_anthropic()

    mock_block = MagicMock()
    mock_block.text = "ответ от Anthropic"
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    provider.client.messages.create = AsyncMock(return_value=mock_response)

    result = await provider.complete(
        [Message(role="user", content="вопрос")], stream=False
    )
    assert result == "ответ от Anthropic"


@pytest.mark.asyncio
async def test_anthropic_complete_no_stream_block_without_text() -> None:
    provider = _make_anthropic()

    mock_block = MagicMock(spec=[])  # нет атрибута .text
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    provider.client.messages.create = AsyncMock(return_value=mock_response)

    result = await provider.complete([Message(role="user", content="вопрос")])
    assert result == ""


@pytest.mark.asyncio
async def test_anthropic_complete_stream_yields_tokens() -> None:
    provider = _make_anthropic()

    stream_mock = MagicMock()
    stream_mock.text_stream = _AsyncIter(["Hello", " ", "world"])
    provider.client.messages.stream = MagicMock(return_value=_AsyncCM(stream_mock))

    gen = await provider.complete(
        [Message(role="system", content="sys"), Message(role="user", content="q")],
        stream=True,
    )
    tokens = [t async for t in gen]
    assert tokens == ["Hello", " ", "world"]


@pytest.mark.asyncio
async def test_anthropic_complete_passes_system_separately() -> None:
    """system должен уходить в отдельный параметр Anthropic API, не в messages."""
    provider = _make_anthropic()

    mock_block = MagicMock()
    mock_block.text = "ok"
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    provider.client.messages.create = AsyncMock(return_value=mock_response)

    await provider.complete(
        [Message(role="system", content="инструкция"), Message(role="user", content="вопрос")]
    )

    call_kwargs = provider.client.messages.create.call_args.kwargs
    # Anthropic API принимает system= отдельно
    assert call_kwargs.get("system") == "инструкция"
    # В messages — только user-туры
    for m in call_kwargs.get("messages", []):
        assert m["role"] != "system"


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


def _make_ollama(model: str = "llama3.2") -> OllamaProvider:
    provider = OllamaProvider.__new__(OllamaProvider)
    provider.model = model
    provider.client = MagicMock()
    return provider


@pytest.mark.asyncio
async def test_ollama_complete_no_stream_returns_content() -> None:
    provider = _make_ollama()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_openai_response("ответ от Ollama")
    )

    result = await provider.complete([Message(role="user", content="вопрос")])
    assert result == "ответ от Ollama"


@pytest.mark.asyncio
async def test_ollama_complete_stream_yields_tokens() -> None:
    provider = _make_ollama()
    chunks = [_openai_chunk("tok1"), _openai_chunk(None), _openai_chunk("tok2")]
    provider.client.chat.completions.create = AsyncMock(
        return_value=_AsyncIter(chunks)
    )

    gen = await provider.complete([Message(role="user", content="q")], stream=True)
    tokens = [t async for t in gen]
    assert tokens == ["tok1", "tok2"]


@pytest.mark.asyncio
async def test_ollama_complete_passes_model() -> None:
    provider = _make_ollama(model="mistral")
    provider.client.chat.completions.create = AsyncMock(
        return_value=_openai_response("ok")
    )

    await provider.complete([Message(role="user", content="вопрос")])

    call_kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("model") == "mistral"
