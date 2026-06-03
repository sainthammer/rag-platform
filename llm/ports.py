"""Абстрактный интерфейс (port) для любого LLM-провайдера.

Следует паттерну Ports & Adapters: код приложения зависит только от этого
интерфейса, а конкретные реализации (OpenAI, Anthropic, Ollama) живут в
adapters.py и подменяются без изменения бизнес-логики.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from .llm_dataclasses import Message


class LLMProvider(ABC):
    """Базовый класс для всех провайдеров языковых моделей.

    Каждая конкретная реализация обязана переопределить только один метод —
    ``complete``. Остальная инфраструктура (RAGPipeline, тесты) работает
    исключительно через этот контракт.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Отправить список сообщений модели и получить ответ.

        Args:
            messages: История диалога в хронологическом порядке.
                Обычно первым идёт ``role="system"`` с инструкцией,
                последним — ``role="user"`` с вопросом.
            stream: Если ``False`` — дождаться полного ответа и вернуть строку.
                Если ``True`` — вернуть ``AsyncGenerator``, который отдаёт
                токены по мере их генерации (удобно для streaming UI).

        Returns:
            ``str``                     при ``stream=False``.
            ``AsyncGenerator[str, None]`` при ``stream=True``; каждый элемент —
            очередной текстовый фрагмент (chunk) от модели.
        """
        ...
