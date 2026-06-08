"""Evaluation runner — прогон всех TestCase через RAGPipeline с отчётом.

Запускает все тест-кейсы через реальный или mock-пайплайн, вычисляет
RAGAS-метрики, выполняет тест на галлюцинации и сохраняет HTML-отчёт.

Режимы запуска::

    # Полностью offline — mock LLM и mock VectorDB
    PYTHONPATH=. python evaluation/eval_runner.py

    # Реальный Ollama LLM + embeddings (ollama serve должен быть запущен)
    PYTHONPATH=. python evaluation/eval_runner.py --mode ollama

    # Ollama + отправка трейсов в Jaeger (нужен --jaeger или OTEL_... в .env)
    PYTHONPATH=. python evaluation/eval_runner.py --mode ollama --jaeger

    # Изменить имя файла отчёта
    PYTHONPATH=. python evaluation/eval_runner.py --output results/report.html

Что делает:
    1. Загружает все 45 тест-кейсов (30 positive / 10 negative / 5 multi_hop).
    2. Для каждого вызывает RAGPipeline.run_detailed() → answer + retrieved_contexts.
    3. На первых N кейсах запускает RAGAS (faithfulness, relevancy, precision, recall).
    4. Тест на галлюцинации: negative-кейсы → проверяет, что ответ содержит
       фразы типа «не нашёл», «недостаточно информации» и т.д.
    5. Генерирует HTML-отчёт с таблицей вопрос/ответ/score/pass-fail.
    6. Если передан --jaeger, все спаны pipeline.run_detailed() уходят в Jaeger.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import types
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Compatibility shim для ragas 0.4.x — до любого импорта ragas
_VERTEXAI_PATH = "langchain_community.chat_models.vertexai"
if _VERTEXAI_PATH not in sys.modules:
    _stub = types.ModuleType(_VERTEXAI_PATH)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[assignment]
    sys.modules[_VERTEXAI_PATH] = _stub

warnings.filterwarnings(
    "ignore", message=".*LangchainLLMWrapper is deprecated.*", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message=".*LangchainEmbeddingsWrapper is deprecated.*", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message=".*langchain-community.*is being sunset.*", category=DeprecationWarning
)

from evaluation.testcase import TestCase
from evaluation.testcases_dataset import get_testcases
from evaluation.ragas_eval import as_hf_dataset, evaluate_ragas
from llm.pipeline import RAGPipeline
from llm.ports import LLMProvider
from llm.llm_dataclasses import Message
from llm.prompt_templates import STRICT
from vector_store.ports import VectorDB
from vector_store.store_dataclasses import SearchResult


# ---------------------------------------------------------------------------
# Mock-компоненты (работают полностью offline, без каких-либо API)
# ---------------------------------------------------------------------------

class _MockVectorDB(VectorDB):
    """Возвращает фиксированный набор Python-документов для любого запроса.

    Используется в mock-режиме, когда нет реальной векторной базы данных.
    Документы содержат базовую Python-теорию — релевантны для positive-кейсов,
    но не содержат ответов на negative-вопросы (о несуществующих фичах Python).
    """

    _DOCS = [
        "PEP 8 — руководство по стилю написания кода на Python. "
        "Рекомендует 4 пробела для отступов, максимум 79 символов в строке, "
        "имена переменных в snake_case, классов в CamelCase.",
        "Python — интерпретируемый язык с динамической типизацией и автоматическим "
        "управлением памятью (garbage collector). Поддерживает ООП, функциональное "
        "и процедурное программирование.",
        "Декораторы в Python — синтаксический сахар для функций высшего порядка. "
        "Позволяют модифицировать поведение функции или класса без изменения их кода. "
        "Применяются через @decorator перед определением.",
        "list comprehension: [x**2 for x in range(10)] — компактный способ создать список. "
        "dict comprehension: {k: v for k, v in items}. "
        "generator expression: (x for x in range(10)) — ленивый итератор.",
        "Исключения в Python: try/except/else/finally. Базовый класс Exception. "
        "Создание собственных: class MyError(Exception): pass. "
        "raise ValueError('message') для явного вызова.",
    ]

    def add(self, ids, embeddings, documents=None, metadatas=None) -> None:  # noqa: D102
        pass

    def delete(self, ids) -> None:  # noqa: D102
        pass

    def count(self) -> int:  # noqa: D102
        return len(self._DOCS)

    def search(self, query_embedding: list[float], n_results: int = 3) -> SearchResult:
        """Вернуть первые n_results документов из фиксированного набора."""
        docs = self._DOCS[:n_results]
        return SearchResult(
            ids=[str(i) for i in range(len(docs))],
            documents=docs,
            distances=[0.1 * (i + 1) for i in range(len(docs))],
            metadatas=[{"source": "mock"} for _ in docs],
        )


class _MockLLMProvider(LLMProvider):
    """LLM-заглушка: всегда возвращает уверенный ответ.

    Специально имитирует галлюцинацию: не проверяет, содержит ли контекст
    ответ на вопрос, и всегда отвечает уверенно. Это значит, что negative-кейсы
    (вопросы о несуществующих функциях Python) провалят тест на галлюцинации —
    именно так и должно работать в демонстрационных целях.

    В real-режиме (Ollama) с шаблоном STRICT модель должна сказать
    «недостаточно информации» для negative-вопросов — и тест пройдёт.
    """

    model = "mock-llm-v1"

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str:
        """Вернуть шаблонный уверенный ответ."""
        # Извлекаем вопрос из последнего user-сообщения
        user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        # Возвращаем уверенный ответ (намеренно галлюцинируем для демонстрации)
        return (
            f"Согласно документации Python, данная возможность реализована "
            f"во встроенных механизмах языка начиная с версии 3.x. "
            f"[Mock-ответ на: {user_msg[:60]}...]"
        )


def _mock_embed_fn(text: str) -> list[float]:
    """Возвращает нулевой вектор — достаточно для работы с _MockVectorDB."""
    return [0.0] * 16


# ---------------------------------------------------------------------------
# Сборка пайплайнов
# ---------------------------------------------------------------------------

def _build_mock_pipeline() -> RAGPipeline:
    """Собрать полностью offline pipeline (mock LLM + mock VectorDB).

    Не требует запущенных внешних сервисов. Используется для проверки
    логики eval_runner'а и HTML-отчёта без реального вызова API.
    """
    return RAGPipeline(
        llm=_MockLLMProvider(),
        vector_db=_MockVectorDB(),
        embed_fn=_mock_embed_fn,
        template=STRICT,
        n_results=3,
    )


def _build_ollama_pipeline() -> RAGPipeline:
    """Собрать pipeline с реальным Ollama LLM и embeddings + mock VectorDB.

    Требует запущенного ``ollama serve``. Для embeddings нужна модель
    ``nomic-embed-text``, для LLM — ``llama3.2`` (или настроенная в .env).

    MockVectorDB используется намеренно, чтобы runner работал без реальных
    данных в ChromaDB/Qdrant. Чанки — фиктивные Python-документы.
    """
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Установите langchain-ollama: pip install langchain-ollama"
        ) from exc

    from llm.adapters import OllamaProvider

    embed_model = OllamaEmbeddings(model="nomic-embed-text")

    def _embed_fn(text: str) -> list[float]:
        return embed_model.embed_query(text)

    return RAGPipeline(
        llm=OllamaProvider(model="llama3.2"),
        vector_db=_MockVectorDB(),
        embed_fn=_embed_fn,
        template=STRICT,
        n_results=3,
    )


def _build_ollama_ragas():
    """Собрать LLM и embeddings для RAGAS в Ollama-режиме.

    Возвращает кортеж (ragas_llm, ragas_embeddings).
    """
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = LangchainLLMWrapper(ChatOllama(model="mistral:7b"))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    return llm, embeddings


# ---------------------------------------------------------------------------
# Тест на галлюцинации
# ---------------------------------------------------------------------------

# Фразы, которые модель должна говорить, когда ответа нет в контексте.
# Шаблон STRICT явно инструктирует модель использовать подобные формулировки.
_NOT_FOUND_PHRASES = [
    "не нашёл", "не могу найти", "не содержит", "нет информации",
    "не знаю", "не упомянут", "недостаточно информации", "нет данных",
    "не могу ответить", "у меня нет", "отсутствует", "не нашли",
    "нет в контексте", "не указан", "нет такого", "не существует",
    "i don't", "not found", "no information", "cannot find",
    "insufficient", "not available", "not mentioned",
]


def _is_not_found(answer: str) -> bool:
    """Проверить, содержит ли ответ отказную фразу типа «не нашёл»."""
    low = answer.lower()
    return any(phrase in low for phrase in _NOT_FOUND_PHRASES)


# ---------------------------------------------------------------------------
# Порог pass/fail для RAGAS-метрик
# ---------------------------------------------------------------------------

_PASS_THRESHOLD = 0.5  # минимальный балл для PASS

_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def _score_class(score: float | None) -> str:
    """Вернуть CSS-класс для цвета оценки."""
    if score is None:
        return "score-na"
    if score >= 0.7:
        return "score-high"
    if score >= 0.4:
        return "score-mid"
    return "score-low"


def _fmt_score(score: float | None) -> str:
    """Отформатировать оценку для вывода."""
    return "—" if score is None else f"{score:.2f}"


# ---------------------------------------------------------------------------
# Основной runner
# ---------------------------------------------------------------------------

async def run_evaluation(
    pipeline: RAGPipeline,
    testcases: list[TestCase],
    ragas_llm: Any = None,
    ragas_embeddings: Any = None,
    max_ragas_cases: int = 5,
) -> dict[str, Any]:
    """Прогнать все тест-кейсы через pipeline и собрать результаты.

    Args:
        pipeline: Собранный RAGPipeline (mock или real).
        testcases: Список всех тест-кейсов для прогона.
        ragas_llm: LLM для RAGAS (None — используются дефолты ragas, обычно OpenAI).
        ragas_embeddings: Embeddings для RAGAS (None — дефолты ragas).
        max_ragas_cases: Число кейсов для RAGAS (полная оценка медленная).

    Returns:
        Словарь с ключами:
            ``rows``                 — все результаты (вопрос/ответ/контексты)
            ``ragas_df``             — DataFrame с per-case RAGAS-баллами или None
            ``ragas_avg``            — средние RAGAS-баллы или None
            ``hallucination_results``— результаты теста на галлюцинации
    """
    rows: list[dict[str, Any]] = []

    print(f"\n{'─'*60}")
    print(f"  Прогон pipeline: {len(testcases)} кейсов")
    print(f"{'─'*60}")

    for i, tc in enumerate(testcases, 1):
        print(f"  [{i:2d}/{len(testcases)}] {tc.category:10} │ {tc.question[:55]}…")
        t0 = time.perf_counter()
        try:
            answer, contexts = await pipeline.run_detailed(tc.question)
            latency_ms = round((time.perf_counter() - t0) * 1000)
            print(f"             └─ OK  {latency_ms:4d}ms │ {len(contexts)} чанков")
        except Exception as exc:
            answer = ""
            contexts = []
            print(f"             └─ ERR │ {exc}")

        rows.append({
            "question": tc.question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": tc.ground_truth,
            "category": tc.category,
        })

    # -- RAGAS ----------------------------------------------------------------
    ragas_df = None
    ragas_avg: dict[str, float | None] = {m: None for m in _METRIC_NAMES}

    ragas_rows = rows[:max_ragas_cases]
    if ragas_llm is not None:
        print(f"\n{'─'*60}")
        print(f"  RAGAS: оцениваем первые {len(ragas_rows)} кейсов…")
        print(f"{'─'*60}")
        try:
            dataset = as_hf_dataset(ragas_rows)
            result = evaluate_ragas(dataset, llm=ragas_llm, embeddings=ragas_embeddings)
            ragas_df = result.to_pandas()
            for m in _METRIC_NAMES:
                if m in ragas_df.columns:
                    ragas_avg[m] = float(ragas_df[m].mean(skipna=True))
            print("  RAGAS завершён.")
        except Exception as exc:
            print(f"  RAGAS ошибка: {exc}")
    else:
        print("\n  RAGAS пропущен (mock-режим не имеет LLM-оценщика).")
        print("  Запустите с --mode ollama для реальных баллов.")

    # -- Тест на галлюцинации ------------------------------------------------
    print(f"\n{'─'*60}")
    print("  Тест на галлюцинации (negative-кейсы):")
    print(f"{'─'*60}")

    hallucination_results: list[dict[str, Any]] = []
    for row in rows:
        if row["category"] != "negative":
            continue
        passed = _is_not_found(row["answer"])
        hallucination_results.append({
            "question": row["question"],
            "answer": row["answer"],
            "passed": passed,
        })
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] {row['question'][:55]}…")

    n_pass = sum(1 for r in hallucination_results if r["passed"])
    n_total = len(hallucination_results)
    print(f"\n  Итог галлюцинаций: {n_pass}/{n_total} PASS")

    return {
        "rows": rows,
        "ragas_df": ragas_df,
        "ragas_avg": ragas_avg,
        "hallucination_results": hallucination_results,
    }


# ---------------------------------------------------------------------------
# HTML-отчёт
# ---------------------------------------------------------------------------

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f0f2f5; color: #1a1a2e; font-size: 14px; line-height: 1.5;
}
.container { max-width: 1500px; margin: 0 auto; padding: 24px; }

/* Header */
.header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
  color: white; padding: 32px 36px; border-radius: 12px; margin-bottom: 24px;
}
.header h1 { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
.header .subtitle { color: #94a3b8; margin-top: 6px; font-size: 13px; }
.header .meta { display: flex; gap: 24px; margin-top: 16px; }
.header .meta-item { background: rgba(255,255,255,0.08); border-radius: 8px;
  padding: 8px 16px; font-size: 12px; }
.header .meta-item strong { color: #e2e8f0; display: block; }
.header .meta-item span { color: #94a3b8; }

/* Cards */
.card {
  background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
}
.card h2 {
  font-size: 16px; font-weight: 700; margin-bottom: 20px; color: #0f172a;
  border-bottom: 2px solid #e2e8f0; padding-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}

/* Stats row */
.stats { display: flex; gap: 16px; flex-wrap: wrap; }
.stat {
  flex: 1; min-width: 120px; background: #f8fafc; border-radius: 10px;
  padding: 16px 20px; border: 1px solid #e2e8f0; text-align: center;
}
.stat .value { font-size: 36px; font-weight: 800; color: #1a1a2e; line-height: 1; }
.stat .label { font-size: 11px; color: #64748b; margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }

/* RAGAS metric bars */
.metric-row {
  display: flex; align-items: center; gap: 16px; padding: 10px 0;
  border-bottom: 1px solid #f1f5f9;
}
.metric-row:last-child { border-bottom: none; }
.metric-name { width: 190px; font-size: 13px; color: #475569; font-weight: 500; }
.metric-value { width: 44px; font-size: 18px; font-weight: 700; text-align: right; }
.metric-bar { flex: 1; background: #e2e8f0; border-radius: 6px; height: 10px;
  overflow: hidden; }
.metric-bar-fill { height: 100%; border-radius: 6px;
  transition: width 0.3s ease; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th {
  background: #f1f5f9; padding: 10px 14px; text-align: left;
  font-weight: 600; color: #475569; border-bottom: 2px solid #e2e8f0;
  white-space: nowrap;
}
td { padding: 10px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }

/* Badges */
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 700; white-space: nowrap;
}
.badge-pass { background: #dcfce7; color: #15803d; }
.badge-fail { background: #fee2e2; color: #b91c1c; }
.badge-na   { background: #f1f5f9; color: #64748b; }
.badge-positive { background: #dbeafe; color: #1d4ed8; }
.badge-negative { background: #fef3c7; color: #b45309; }
.badge-multi_hop { background: #f3e8ff; color: #7e22ce; }

/* Score colors */
.score-high { color: #15803d; font-weight: 700; }
.score-mid  { color: #b45309; font-weight: 700; }
.score-low  { color: #b91c1c; font-weight: 700; }
.score-na   { color: #94a3b8; }

/* Text cells */
.cell-question { max-width: 220px; }
.cell-answer   { max-width: 280px; }
.cell-gt       { max-width: 200px; color: #64748b; font-style: italic; }
.truncate { overflow: hidden; display: -webkit-box; -webkit-line-clamp: 4;
  -webkit-box-orient: vertical; }

/* Hallucination section */
.hall-note {
  background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
  padding: 12px 16px; margin-bottom: 16px; font-size: 12.5px; color: #78350f;
}

/* Footer */
.footer { text-align: center; color: #94a3b8; font-size: 11px; margin-top: 32px; padding: 16px; }
"""


def _bar_color(score: float) -> str:
    """Вернуть цвет заливки прогресс-бара в зависимости от балла."""
    if score >= 0.7:
        return "#22c55e"   # зелёный
    if score >= 0.4:
        return "#f59e0b"   # жёлтый
    return "#ef4444"       # красный


def _category_badge(cat: str) -> str:
    """Вернуть HTML-бейдж для категории тест-кейса."""
    label = {"positive": "positive", "negative": "negative", "multi_hop": "multi-hop"}.get(
        cat, cat
    )
    return f'<span class="badge badge-{cat}">{label}</span>'


def _pass_badge(passed: bool) -> str:
    """Вернуть HTML-бейдж PASS/FAIL."""
    cls = "badge-pass" if passed else "badge-fail"
    txt = "PASS" if passed else "FAIL"
    return f'<span class="badge {cls}">{txt}</span>'


def _esc(text: str) -> str:
    """Экранировать HTML-спецсимволы в тексте."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_html_report(
    result: dict[str, Any],
    mode: str,
    output_path: str,
    max_ragas_cases: int,
) -> None:
    """Сгенерировать и сохранить HTML-отчёт с результатами оценки.

    Args:
        result: Словарь из ``run_evaluation()`` с ключами rows, ragas_df,
            ragas_avg, hallucination_results.
        mode: Строка-метка режима (``"mock"`` или ``"ollama"``).
        output_path: Путь к выходному HTML-файлу.
        max_ragas_cases: Сколько кейсов оценивалось через RAGAS.
    """
    rows = result["rows"]
    ragas_df = result["ragas_df"]
    ragas_avg = result["ragas_avg"]
    hall_results = result["hallucination_results"]

    n_total = len(rows)
    n_negative = sum(1 for r in rows if r["category"] == "negative")
    n_hall_pass = sum(1 for r in hall_results if r["passed"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Определяем pass/fail для каждого row (по RAGAS если есть)
    row_scores: list[dict[str, float | None]] = []
    for i, row in enumerate(rows):
        scores: dict[str, float | None] = {m: None for m in _METRIC_NAMES}
        if ragas_df is not None and i < len(ragas_df):
            for m in _METRIC_NAMES:
                if m in ragas_df.columns:
                    v = ragas_df.iloc[i][m]
                    scores[m] = None if (v is None or str(v) == "nan") else float(v)
        row_scores.append(scores)

    # --- HTML-блок: секция RAGAS ---
    ragas_section = ""
    if any(v is not None for v in ragas_avg.values()):
        bars_html = ""
        for metric in _METRIC_NAMES:
            score = ragas_avg[metric]
            if score is None:
                bars_html += f"""
            <div class="metric-row">
              <div class="metric-name">{metric}</div>
              <div class="metric-value score-na">—</div>
              <div class="metric-bar"><div class="metric-bar-fill" style="width:0%;background:#e2e8f0"></div></div>
            </div>"""
            else:
                pct = int(score * 100)
                color = _bar_color(score)
                css = _score_class(score)
                bars_html += f"""
            <div class="metric-row">
              <div class="metric-name">{metric}</div>
              <div class="metric-value {css}">{score:.2f}</div>
              <div class="metric-bar">
                <div class="metric-bar-fill" style="width:{pct}%;background:{color}"></div>
              </div>
            </div>"""
        ragas_section = f"""
      <div class="card">
        <h2>📊 RAGAS-метрики <small style="font-weight:400;font-size:12px;color:#94a3b8">(первые {max_ragas_cases} кейсов)</small></h2>
        {bars_html}
      </div>"""
    else:
        ragas_section = """
      <div class="card">
        <h2>📊 RAGAS-метрики</h2>
        <p style="color:#94a3b8;font-size:13px">
          RAGAS не запускался в mock-режиме.<br>
          Запустите <code>--mode ollama</code> для реальных баллов.
        </p>
      </div>"""

    # --- HTML-блок: тест на галлюцинации ---
    hall_rows_html = ""
    hall_q_index = 0
    for row in rows:
        if row["category"] != "negative":
            continue
        hr = hall_results[hall_q_index]
        hall_q_index += 1
        badge = _pass_badge(hr["passed"])
        hall_rows_html += f"""
          <tr>
            <td>{badge}</td>
            <td class="cell-question"><div class="truncate">{_esc(row['question'])}</div></td>
            <td class="cell-answer"><div class="truncate">{_esc(row['answer'])}</div></td>
          </tr>"""

    # --- HTML-блок: детальная таблица ---
    detail_rows_html = ""
    hall_q_idx2 = 0
    for i, row in enumerate(rows):
        s = row_scores[i]
        cat = row["category"]

        if cat == "negative":
            hr = hall_results[hall_q_idx2]
            hall_q_idx2 += 1
            overall = _pass_badge(hr["passed"])
        else:
            vals = [v for v in s.values() if v is not None]
            if not vals:
                overall = '<span class="badge badge-na">—</span>'
            else:
                overall = _pass_badge(all(v >= _PASS_THRESHOLD for v in vals))

        score_cells = ""
        for m in _METRIC_NAMES:
            v = s[m]
            css = _score_class(v)
            score_cells += f'<td class="{css}">{_fmt_score(v)}</td>'

        gt_text = _esc(row["ground_truth"][:120] + "…") if row["ground_truth"] else '<span class="score-na">(пусто — нет правильного ответа)</span>'
        detail_rows_html += f"""
          <tr>
            <td style="color:#94a3b8;text-align:center">{i+1}</td>
            <td>{_category_badge(cat)}</td>
            <td class="cell-question"><div class="truncate">{_esc(row['question'])}</div></td>
            <td class="cell-answer"><div class="truncate">{_esc(row['answer'][:300])}</div></td>
            <td class="cell-gt"><div class="truncate">{gt_text}</div></td>
            {score_cells}
            <td>{overall}</td>
          </tr>"""

    # --- Итоговый HTML ---
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Evaluation Report — {now}</title>
  <style>{_HTML_CSS}</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>RAG Platform — Evaluation Report</h1>
    <div class="subtitle">Автоматическая оценка качества RAG-пайплайна</div>
    <div class="meta">
      <div class="meta-item"><strong>{now}</strong><span>дата и время</span></div>
      <div class="meta-item"><strong>{mode}</strong><span>режим</span></div>
      <div class="meta-item"><strong>{n_total}</strong><span>всего кейсов</span></div>
      <div class="meta-item"><strong>{max_ragas_cases}</strong><span>RAGAS кейсов</span></div>
    </div>
  </div>

  <!-- Summary -->
  <div class="card">
    <h2>📋 Сводка</h2>
    <div class="stats">
      <div class="stat">
        <div class="value">{n_total}</div>
        <div class="label">Всего кейсов</div>
      </div>
      <div class="stat">
        <div class="value">{sum(1 for r in rows if r['category']=='positive')}</div>
        <div class="label">Positive</div>
      </div>
      <div class="stat">
        <div class="value">{n_negative}</div>
        <div class="label">Negative</div>
      </div>
      <div class="stat">
        <div class="value">{sum(1 for r in rows if r['category']=='multi_hop')}</div>
        <div class="label">Multi-hop</div>
      </div>
      <div class="stat">
        <div class="value" style="color:{'#15803d' if n_hall_pass==n_negative else '#b91c1c'}">{n_hall_pass}/{n_negative}</div>
        <div class="label">Галлюцинации PASS</div>
      </div>
    </div>
  </div>

  <!-- RAGAS metrics -->
  {ragas_section}

  <!-- Hallucination test -->
  <div class="card">
    <h2>🔍 Тест на галлюцинации</h2>
    <div class="hall-note">
      <strong>Что тестируется:</strong> negative-кейсы содержат вопросы о несуществующих
      возможностях Python (например, «Какой встроенный модуль обеспечивает доступ к GPU через CUDA?»).
      Правильный ответ — отказ: модель должна сказать «не нашёл», «недостаточно информации» и т.п.<br>
      <strong>FAIL</strong> означает, что модель галлюцинировала — придумала несуществующий ответ.
      В mock-режиме это ожидаемо, в real-режиме (Ollama + STRICT prompt) модель должна отказывать.
    </div>
    <table>
      <thead>
        <tr><th>Статус</th><th>Вопрос</th><th>Ответ модели</th></tr>
      </thead>
      <tbody>{hall_rows_html}</tbody>
    </table>
  </div>

  <!-- Detailed results -->
  <div class="card">
    <h2>📑 Детальные результаты всех кейсов</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Тип</th>
          <th>Вопрос</th>
          <th>Ответ</th>
          <th>Эталон (ground truth)</th>
          <th>Faith.</th>
          <th>Relev.</th>
          <th>Prec.</th>
          <th>Recall</th>
          <th>Итог</th>
        </tr>
      </thead>
      <tbody>{detail_rows_html}</tbody>
    </table>
  </div>

  <div class="footer">
    Сгенерировано eval_runner.py · RAG Platform · {now}
  </div>

</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI-точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа командной строки eval_runner.

    Парсит аргументы, строит pipeline, запускает оценку и сохраняет отчёт.
    """
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Runner — прогон тест-кейсов и генерация отчёта",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python evaluation/eval_runner.py                       # mock, быстро, offline
  python evaluation/eval_runner.py --mode ollama         # Ollama LLM + RAGAS
  python evaluation/eval_runner.py --mode ollama --jaeger  # + трейсы в Jaeger
  python evaluation/eval_runner.py --max-cases 10        # RAGAS на 10 кейсах
  python evaluation/eval_runner.py --output results/report.html
""",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "ollama"],
        default="mock",
        help="mock = полностью offline, ollama = реальный Ollama LLM (default: mock)",
    )
    parser.add_argument(
        "--output",
        default="eval_report.html",
        help="Путь к HTML-отчёту (default: eval_report.html)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=5,
        metavar="N",
        help="Число кейсов для RAGAS-оценки — больше = точнее, но медленнее (default: 5)",
    )
    parser.add_argument(
        "--jaeger",
        action="store_true",
        help="Отправлять OTel-трейсы в Jaeger (нужен OTEL_EXPORTER_OTLP_ENDPOINT)",
    )
    args = parser.parse_args()

    # Настройка трейсинга (если запрошено)
    otel_provider = None
    if args.jaeger:
        from observability.tracing import setup_tracing
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        print(f"OTel → Jaeger: {endpoint}")
        otel_provider = setup_tracing("rag-eval-runner")

    print(f"\n{'═'*60}")
    print(f"  RAG Evaluation Runner")
    print(f"  Режим:  {args.mode}")
    print(f"  Отчёт: {args.output}")
    print(f"{'═'*60}")

    # Сборка pipeline
    print("\n  Собираем pipeline…")
    if args.mode == "ollama":
        pipeline = _build_ollama_pipeline()
        ragas_llm, ragas_embeddings = _build_ollama_ragas()
        print("  Pipeline: Ollama LLM + Ollama Embeddings + MockVectorDB")
    else:
        pipeline = _build_mock_pipeline()
        ragas_llm = ragas_embeddings = None
        print("  Pipeline: MockLLM + MockVectorDB (полностью offline)")

    # Загружаем все тест-кейсы
    testcases = get_testcases()
    print(f"  Тест-кейсов: {len(testcases)}")

    # Запускаем
    result = asyncio.run(
        run_evaluation(
            pipeline=pipeline,
            testcases=testcases,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            max_ragas_cases=args.max_cases,
        )
    )

    # Генерируем отчёт
    print(f"\n  Генерируем HTML-отчёт…")
    generate_html_report(
        result=result,
        mode=args.mode,
        output_path=args.output,
        max_ragas_cases=args.max_cases,
    )

    n_hall = len(result["hallucination_results"])
    n_pass = sum(1 for r in result["hallucination_results"] if r["passed"])
    ragas_avg = result["ragas_avg"]

    print(f"\n{'═'*60}")
    print(f"  Готово!")
    print(f"  Отчёт сохранён: {Path(args.output).resolve()}")
    print(f"  Галлюцинации:   {n_pass}/{n_hall} PASS")
    if any(v is not None for v in ragas_avg.values()):
        print(f"  RAGAS (avg):")
        for m, v in ragas_avg.items():
            bar = "█" * int((v or 0) * 10) + "░" * (10 - int((v or 0) * 10))
            print(f"    {m:22}: {_fmt_score(v)}  {bar}")
    print(f"{'═'*60}\n")

    # Сбрасываем буферизованные OTel-спаны в Jaeger перед выходом.
    # BatchSpanProcessor отправляет спаны асинхронно пачками — без явного
    # flush/shutdown они теряются при os._exit(0).
    if otel_provider is not None:
        otel_provider.force_flush()
        otel_provider.shutdown()

    sys.stdout.flush()
    os._exit(0)  # обходит multiprocess.ResourceTracker bug на Python 3.12 Windows


if __name__ == "__main__":
    main()
