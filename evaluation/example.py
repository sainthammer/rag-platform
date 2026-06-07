"""Демонстрация модуля evaluation: тест-кейсы и RAGAS-оценка.

Запуск:
    PYTHONPATH=. .venv/Scripts/python evaluation/example.py
    PYTHONPATH=. .venv/Scripts/python evaluation/example.py ollama   # без API-ключа

Что демонстрируется:
    1. Загрузка фиксированного датасета из 45 тест-кейсов и статистика по категориям.
    2. Структура TestCase и обращение к его полям.
    3. Построение RAGAS-строк (build_rows_from_testcases) с mock-ответами и контекстами.
    4. Сборка HuggingFace Dataset (as_hf_dataset) — требует пакета 'datasets'.
    5. Запуск RAGAS-оценки через OpenAI (требует OPENAI_API_KEY)
       или через Ollama (требует запущенного сервера, ключ не нужен).

Установка зависимостей:
    pip install -e '.[eval]'
    pip install langchain-ollama   # только для режима Ollama
"""

import os
import sys
import warnings
from collections import Counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LangchainLLMWrapper / LangchainEmbeddingsWrapper помечены deprecated в ragas 0.4.x,
# но llm_factory (рекомендованная замена) несовместима с Ollama — отправляет JSON Schema
# через instructor, которую Ollama не принимает (400 invalid input type).
# Используем Langchain-обёртки как единственный рабочий вариант для локальных моделей.
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangchainEmbeddingsWrapper is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-community.*is being sunset.*", category=DeprecationWarning)

from evaluation.testcase import TestCase
from evaluation.testcases_dataset import get_testcases
from evaluation.ragas_eval import build_rows_from_testcases, as_hf_dataset, evaluate_ragas


# ---------------------------------------------------------------------------
# Фиктивные ответы и контексты для демонстрации без реального RAG-пайплайна
# ---------------------------------------------------------------------------

_MOCK_ANSWER = "Это демонстрационный ответ, сгенерированный mock-провайдером."
_MOCK_CONTEXT = [
    "Python — высокоуровневый язык программирования общего назначения.",
    "PEP 8 задаёт стандарт оформления кода на Python.",
]


# ---------------------------------------------------------------------------
# 1. Загрузка тест-кейсов
# ---------------------------------------------------------------------------

def demo_load_testcases() -> list[TestCase]:
    """Загрузить датасет и вывести статистику."""
    testcases = get_testcases()
    counts = Counter(tc.category for tc in testcases)

    print("─── 1. Датасет тест-кейсов ───")
    print(f"  Всего кейсов : {len(testcases)}")
    for category, count in sorted(counts.items()):
        print(f"  {category:10}: {count}")
    print()
    return testcases


# ---------------------------------------------------------------------------
# 2. Структура TestCase
# ---------------------------------------------------------------------------

def demo_testcase_structure(testcases: list[TestCase]) -> None:
    """Показать структуру первого и первого negative-кейса."""
    positive = next(tc for tc in testcases if tc.category == "positive")
    negative = next(tc for tc in testcases if tc.category == "negative")
    multi_hop = next(tc for tc in testcases if tc.category == "multi_hop")

    print("─── 2. Примеры TestCase ───")
    for label, tc in [("positive", positive), ("negative", negative), ("multi_hop", multi_hop)]:
        print(f"  [{label}]")
        print(f"    question    : {tc.question[:70]}...")
        print(f"    ground_truth: {tc.ground_truth[:60]}..." if tc.ground_truth else "    ground_truth: (пусто — нет правильного ответа)")
        print(f"    category    : {tc.category}")
    print()


# ---------------------------------------------------------------------------
# 3. Построение строк для RAGAS
# ---------------------------------------------------------------------------

def demo_build_rows(testcases: list[TestCase]) -> list[dict]:
    """Преобразовать тест-кейсы в строки для RAGAS с mock-данными."""
    sample = testcases[:5]

    answer_by_question = {tc.question: _MOCK_ANSWER for tc in sample}
    contexts_by_question = {tc.question: _MOCK_CONTEXT for tc in sample}

    rows = build_rows_from_testcases(
        sample,
        answer_by_question=answer_by_question,
        contexts_by_question=contexts_by_question,
    )

    print("─── 3. build_rows_from_testcases() ───")
    print(f"  Построено строк: {len(rows)}")
    print(f"  Ключи строки   : {list(rows[0].keys())}")
    print(f"  Пример строки  :")
    row = rows[0]
    print(f"    question    : {row['question'][:60]}...")
    print(f"    answer      : {row['answer']}")
    print(f"    contexts    : {row['contexts']}")
    print(f"    ground_truth: {row['ground_truth'][:60]}...")
    print(f"    category    : {row['category']}")
    print()
    return rows


# ---------------------------------------------------------------------------
# 4. Сборка HuggingFace Dataset
# ---------------------------------------------------------------------------

def demo_hf_dataset(rows: list[dict]) -> None:
    """Собрать HuggingFace Dataset из строк."""
    print("─── 4. as_hf_dataset() ───")
    try:
        dataset = as_hf_dataset(rows)
        print(f"  Dataset создан: {dataset}")
        print(f"  Колонки: {dataset.column_names}")
        print(f"  Строк  : {len(dataset)}")
    except ImportError as exc:
        print(f"  Пропущено: {exc}")
    print()


# ---------------------------------------------------------------------------
# 5a. RAGAS-оценка через OpenAI
# ---------------------------------------------------------------------------

def demo_evaluate_openai(rows: list[dict]) -> None:
    """Запустить RAGAS-оценку через OpenAI (требует OPENAI_API_KEY)."""
    print("─── 5a. evaluate_ragas() via OpenAI ───")

    if not os.getenv("OPENAI_API_KEY"):
        print("  Пропущено: OPENAI_API_KEY не задан.")
        print("  Задайте ключ в .env или запустите с аргументом 'ollama'.\n")
        return

    try:
        dataset = as_hf_dataset(rows)
        result = evaluate_ragas(dataset)
        avg = result.to_pandas().mean(numeric_only=True)
        print("  Результаты RAGAS:")
        for metric, score in avg.items():
            print(f"    {metric:25}: {score:.4f}")
    except ImportError as exc:
        print(f"  Пропущено: {exc}")
    print()


# ---------------------------------------------------------------------------
# 5b. RAGAS-оценка через Ollama (без API-ключа)
# ---------------------------------------------------------------------------

def _build_ollama_llm(model: str = "mistral:7b"):
    """Собрать ragas LLM через ChatOllama (langchain_ollama).

    llm_factory нельзя использовать с Ollama — он использует библиотеку instructor,
    которая отправляет JSON Schema в response_format. Ollama это не принимает (400).
    ChatOllama через LangchainLLMWrapper корректно форматирует JSON под Ollama.
    """
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatOllama(model=model))


def _build_ollama_embeddings(model: str = "nomic-embed-text"):
    """Собрать ragas Embeddings через OllamaEmbeddings (langchain_ollama)."""
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(OllamaEmbeddings(model=model))


def demo_evaluate_ollama(rows: list[dict]) -> None:
    """Запустить RAGAS-оценку через локальный Ollama (ключ не нужен).

    Требования:
        ollama serve
        ollama pull llama3.2
        ollama pull nomic-embed-text
    """
    print("─── 5b. evaluate_ragas() via Ollama ───")
    print("  (локальная модель работает медленнее облачных API, оценка займёт 1-3 мин)")

    try:
        llm = _build_ollama_llm()
        embeddings = _build_ollama_embeddings()
        dataset = as_hf_dataset(rows)
        result = evaluate_ragas(dataset, llm=llm, embeddings=embeddings)
        df = result.to_pandas()
        # TimeoutError в отдельных job-ах — нормально для локальных моделей:
        # ragas запускает задачи асинхронно с таймаутом; медленные запросы отбрасываются,
        # остальные усредняются. NaN исключаются из mean автоматически.
        avg = df.mean(numeric_only=True)
        n_ok = df.notna().all(axis=1).sum()
        print(f"  Оценено кейсов: {n_ok}/{len(df)} (остальные — timeout)")
        print("  Результаты RAGAS (Ollama):")
        for metric, score in avg.items():
            print(f"    {metric:25}: {score:.4f}")
    except ImportError as exc:
        print(f"  Пропущено: {exc}")
        print("  Установите: pip install langchain-ollama")
    except Exception as exc:
        print(f"  Ошибка: {exc}")
        print("  Убедитесь, что Ollama запущен: ollama serve")
    print()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    use_ollama = len(sys.argv) > 1 and sys.argv[1] == "ollama"

    testcases = demo_load_testcases()
    demo_testcase_structure(testcases)
    rows = demo_build_rows(testcases)
    demo_hf_dataset(rows)

    if use_ollama:
        demo_evaluate_ollama(rows)
    else:
        demo_evaluate_openai(rows)

    print("✓ Готово")
    sys.stdout.flush()
    # os._exit(0) обходит деструкторы multiprocess.ResourceTracker, которые падают
    # на Python 3.12 Windows из-за бага в multiprocess==0.70.x (нет _recursion_count у RLock)
    os._exit(0)


if __name__ == "__main__":
    main()
