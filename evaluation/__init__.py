"""Пакет evaluation: оценка RAG-пайплайна через RAGAS.

Экспортирует:
    TestCase                  — датакласс тест-кейса.
    get_testcases             — фиксированный датасет из 45 кейсов.
    build_rows_from_testcases — преобразование TestCase → строки для RAGAS.
    as_hf_dataset             — сборка HuggingFace Dataset.
    evaluate_ragas            — запуск оценки по метрикам RAGAS.
"""

from evaluation.ragas_eval import as_hf_dataset, build_rows_from_testcases, evaluate_ragas
from evaluation.testcase import TestCase
from evaluation.testcases_dataset import get_testcases

__all__ = [
    "TestCase",
    "get_testcases",
    "as_hf_dataset",
    "build_rows_from_testcases",
    "evaluate_ragas",
]

