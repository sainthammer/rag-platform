"""Общие структуры данных для LLM-модуля."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    """Одно сообщение в диалоге с языковой моделью.

    Attributes:
        role: Роль отправителя сообщения.
            - ``"system"``    — системная инструкция (поведение модели, контекст RAG).
            - ``"user"``      — запрос пользователя.
            - ``"assistant"`` — ответ модели (нужен при multi-turn диалоге).
        content: Текст сообщения.
    """

    role: Literal["system", "user", "assistant"]
    content: str
