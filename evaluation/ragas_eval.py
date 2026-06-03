from __future__ import annotations

from typing import Any, TYPE_CHECKING

from evaluation.testcase import TestCase

if TYPE_CHECKING:
    from datasets import Dataset


def as_hf_dataset(
    rows: list[dict[str, Any]],
) -> Dataset:
    """Собирает HuggingFace `Dataset` в минимальном формате, ожидаемом RAGAS.

    Ожидаемые ключи в каждой строке:
    - `question`: str
    - `answer`: str
    - `contexts`: list[str]
    - `ground_truth`: str

    Дополнительные ключи допускаются (например, `source_doc`, `category`).
    """

    try:
        from datasets import Dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Missing optional dependency 'datasets'. Install with: pip install -e '.[eval]'"
        ) from e

    return Dataset.from_list(rows)


def build_rows_from_testcases(
    testcases: list[TestCase],
    *,
    answer_by_question: dict[str, str] | None = None,
    contexts_by_question: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Преобразует список `TestCase` в список строк для RAGAS.

    `answer_by_question` и `contexts_by_question` ожидаются в виде мап:
    - ключ: точный текст вопроса
    - значение: ответ модели / список retrieved-контекстов

    Если для вопроса нет ответа/контекста, подставляются пустые значения.
    """

    answer_by_question = answer_by_question or {}
    contexts_by_question = contexts_by_question or {}

    rows: list[dict[str, Any]] = []
    for tc in testcases:
        rows.append(
            {
                "question": tc.question,
                "ground_truth": tc.ground_truth,
                "answer": answer_by_question.get(tc.question, ""),
                "contexts": contexts_by_question.get(tc.question, []),
                "source_doc": tc.source_doc,
                "category": tc.category,
            }
        )
    return rows


def evaluate_ragas(dataset: "Dataset"):
    """Запускает RAGAS-оценку на датасете.

    Используемые метрики:
    - `faithfulness`
    - `answer_relevancy`
    - `context_recall`
    - `context_precision`
    """

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Missing optional dependency 'ragas'. Install with: pip install -e '.[eval]'"
        ) from e

    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
