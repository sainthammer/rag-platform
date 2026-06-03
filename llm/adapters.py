"""Конкретные реализации LLMProvider для OpenAI, Anthropic и Ollama.

Все три класса следуют одному контракту (LLMProvider.complete) и различаются
только деталями SDK и особенностями протокола каждого провайдера.

Импорты SDK (openai, anthropic) выполняются лениво внутри __init__, чтобы
приложение не падало при старте, если какой-то пакет не установлен.
"""

from typing import AsyncGenerator

from .llm_dataclasses import Message
from .ports import LLMProvider


class OpenAIProvider(LLMProvider):
    """Провайдер для моделей OpenAI.

    Args:
        model: Идентификатор модели OpenAI.
        api_key: API-ключ. Если ``None``, SDK читает переменную окружения
            ``OPENAI_API_KEY``.
        base_url: Базовый URL API. По умолчанию — официальный endpoint OpenAI.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Выполнить запрос к OpenAI Chat Completions API.

        Args:
            messages: История диалога (system → user → …).
            stream: ``True`` — вернуть async-генератор с токенами.

        Returns:
            Полный ответ (``str``) или генератор фрагментов.
        """
        # Конвертируем внутренние Message в формат OpenAI API (list[dict]).
        payload = [{"role": m.role, "content": m.content} for m in messages]

        if not stream:
            # Стандартный blocking-запрос: ждём полного ответа от сервера.
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
            )
            # choices[0] — первый (и обычно единственный) вариант ответа.
            # Fallback на "" защищает от None при пустом ответе модели.
            return response.choices[0].message.content or ""

        # Streaming: возвращаем генератор, не блокируя event loop.
        # Используем create(stream=True) — низкоуровневый SSE-итератор,
        # совместимый с любым OpenAI-совместимым endpoint'ом (в т.ч. Ollama).
        async def _stream() -> AsyncGenerator[str, None]:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
                stream=True,
            )
            async for chunk in response:
                # delta.content может быть None в служебных чанках (первый/последний).
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _stream()


class AnthropicProvider(LLMProvider):
    """Провайдер для моделей Anthropic Claude.

    Args:
        model: Идентификатор модели (например, ``claude-haiku-4-5-20251001``).
        api_key: API-ключ. Если ``None``, SDK читает ``ANTHROPIC_API_KEY``.
        max_tokens: Максимальное число токенов в ответе. Anthropic API требует
            явного указания этого параметра (в отличие от OpenAI).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(api_key=api_key)

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Разделить список сообщений на системный промпт и диалоговые туры.

        Anthropic API принципиально отличается от OpenAI: системный промпт
        передаётся отдельным параметром ``system``, а не первым элементом
        списка ``messages``. Этот метод выполняет нужное разделение.

        Args:
            messages: Все сообщения, включая возможный system.

        Returns:
            Кортеж (system_content | None, список диалоговых сообщений).
        """
        system: str | None = None
        turns: list[dict[str, str]] = []

        for m in messages:
            if m.role == "system":
                # Берём текст системного сообщения — он уйдёт в параметр system=.
                system = m.content
            else:
                # user и assistant-сообщения образуют диалоговые туры.
                turns.append({"role": m.role, "content": m.content})

        return system, turns

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Выполнить запрос к Anthropic Messages API.

        Args:
            messages: История диалога (system, user, assistant, …).
            stream: ``True`` — вернуть async-генератор с токенами.

        Returns:
            Полный ответ (``str``) или генератор фрагментов.
        """
        system, turns = self._split_messages(messages)

        # Собираем kwargs динамически, т.к. system — необязательный параметр.
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": turns,
        }
        if system:
            kwargs["system"] = system  # передаём system только если он есть

        if not stream:
            response = await self.client.messages.create(**kwargs)
            # response.content — список блоков (TextBlock, ToolUseBlock и др.).
            # Берём первый блок; если у него нет атрибута .text — возвращаем "".
            block = response.content[0]
            return block.text if hasattr(block, "text") else ""

        async def _stream() -> AsyncGenerator[str, None]:
            # messages.stream — SSE-стриминг Anthropic SDK с тем же API,
            # что и у OpenAI: context manager + async-итератор text_stream.
            async with self.client.messages.stream(**kwargs) as s:
                async for text in s.text_stream:
                    yield text

        return _stream()


class OllamaProvider(LLMProvider):
    """Провайдер для локальных моделей через Ollama.

    Ollama предоставляет OpenAI-совместимый REST API (``/v1/chat/completions``),
    поэтому здесь используется тот же ``AsyncOpenAI``-клиент с переопределённым
    ``base_url`` — новые зависимости не нужны.

    Args:
        model: Имя модели в Ollama (например, ``llama3.2``, ``mistral``).
        base_url: URL Ollama-сервера. По умолчанию — локальный стандартный порт.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        # Ollama не проверяет значение api_key, но openai SDK требует,
        # чтобы оно было непустой строкой — передаём произвольное значение.
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Выполнить запрос к Ollama через OpenAI-совместимый API.

        Args:
            messages: История диалога.
            stream: ``True`` — вернуть async-генератор с токенами.

        Returns:
            Полный ответ (``str``) или генератор фрагментов.
        """
        # Логика идентична OpenAIProvider — Ollama полностью совместима.
        payload = [{"role": m.role, "content": m.content} for m in messages]

        if not stream:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
            )
            return response.choices[0].message.content or ""

        async def _stream() -> AsyncGenerator[str, None]:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _stream()
