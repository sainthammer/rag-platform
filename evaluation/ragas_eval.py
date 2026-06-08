"""Утилиты для RAGAS-оценки RAG-пайплайна.

Преобразует список `TestCase` в HuggingFace Dataset и запускает оценку
по метрикам: Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall.

Требует опциональных зависимостей: pip install -e '.[eval]'
"""

from __future__ import annotations

import sys
import types
from typing import Any, TYPE_CHECKING

from evaluation.testcase import TestCase

if TYPE_CHECKING:
    from datasets import Dataset
    from ragas.llms import BaseRagasLLM
    from ragas.embeddings import BaseRagasEmbeddings

# ---------------------------------------------------------------------------
# Compatibility shim для ragas 0.4.x
# ---------------------------------------------------------------------------
# ragas 0.4.x делает жёсткий import ChatVertexAI из langchain_community.chat_models.vertexai,
# который был удалён в langchain_community >= 0.3.x.
# Инжектируем заглушку, чтобы import ragas не падал, когда Vertex AI не используется.
_VERTEXAI_PATH = "langchain_community.chat_models.vertexai"
if _VERTEXAI_PATH not in sys.modules:
    _stub = types.ModuleType(_VERTEXAI_PATH)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[assignment]
    sys.modules[_VERTEXAI_PATH] = _stub


def as_hf_dataset(
    rows: list[dict[str, Any]],
) -> "Dataset":
    """Собирает HuggingFace `Dataset` в минимальном формате.

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
            "Отсутствует опциональная зависимость 'datasets'. Установите след. командой: pip install -e '.[eval]'"
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


def evaluate_ragas(
    dataset: "Dataset",
    llm: "BaseRagasLLM | None" = None,
    embeddings: "BaseRagasEmbeddings | None" = None,
):
    """Запускает RAGAS-оценку (ragas >= 0.4.x) на HuggingFace Dataset.

    Используемые метрики:
    - `Faithfulness`       — верность ответа контексту
    - `ResponseRelevancy`  — релевантность ответа вопросу
    - `ContextPrecision`   — точность retrieved-чанков
    - `ContextRecall`      — полнота retrieved-чанков

    По умолчанию RAGAS использует OpenAI (требует `OPENAI_API_KEY`).
    Чтобы использовать локальную модель (например Ollama), передайте
    `llm` и `embeddings`, собранные через `ragas.llms.LangchainLLMWrapper`
    и `ragas.embeddings.LangchainEmbeddingsWrapper` поверх `langchain_openai.ChatOpenAI`
    с `base_url="http://localhost:11434/v1"` (OpenAI-совместимый API Ollama).
    """

    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            ResponseRelevancy,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Отсутствует опциональная зависимость 'ragas'. Установите след. командой: pip install -e '.[eval]'"
        ) from e

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            reference=row["ground_truth"],
            retrieved_contexts=row["contexts"],
        )
        for row in dataset
    ]

    ragas_dataset = EvaluationDataset(samples=samples)

    kwargs: dict[str, Any] = {}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    return evaluate(
        dataset=ragas_dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()],
        **kwargs,
    )