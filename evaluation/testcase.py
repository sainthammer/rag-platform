from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TestCase:
    """Тестовый кейс для оценки RAG.

    - `question`: вопрос пользователя.
    - `ground_truth`: эталонный ответ (для negative-кейсов может быть пустым).
    - `source_doc`: ссылка на источник в базе знаний (обычно относительный путь/идентификатор).
    - `category`: категория кейса (например: positive/negative/multi_hop).
    """

    question: str
    ground_truth: str
    source_doc: str
    category: str
