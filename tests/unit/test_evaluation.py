"""Unit-тесты для модуля evaluation.

Покрывают:
  - вспомогательные функции eval_runner (без запуска pipeline)
  - mock-компоненты для offline-режима
  - generate_html_report: структура выходного файла
  - ragas_eval: build_rows_from_testcases
  - testcases_dataset: количество и структура кейсов
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.eval_runner import (
    _bar_color,
    _category_badge,
    _esc,
    _fmt_score,
    _is_not_found,
    _mock_embed_fn,
    _pass_badge,
    _score_class,
    generate_html_report,
    run_evaluation,
    _MockLLMProvider,
    _MockVectorDB,
    _build_mock_pipeline,
    _METRIC_NAMES,
    _PASS_THRESHOLD,
)
from evaluation.ragas_eval import build_rows_from_testcases
from evaluation.testcase import TestCase
from evaluation.testcases_dataset import get_testcases


# ---------------------------------------------------------------------------
# _is_not_found
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer,expected", [
    ("Я не нашёл информации по этому вопросу.", True),
    ("К сожалению, нет данных в контексте.", True),
    ("Недостаточно информации для ответа.", True),
    ("I don't have enough information.", True),
    ("Нет в контексте таких сведений.", True),
    ("Python is a high-level programming language.", False),
    ("FastAPI поддерживает асинхронные эндпоинты.", False),
    ("", False),
])
def test_is_not_found(answer: str, expected: bool) -> None:
    assert _is_not_found(answer) is expected


# ---------------------------------------------------------------------------
# _score_class
# ---------------------------------------------------------------------------

def test_score_class_high() -> None:
    assert _score_class(0.9) == "score-high"
    assert _score_class(0.7) == "score-high"


def test_score_class_mid() -> None:
    assert _score_class(0.5) == "score-mid"
    assert _score_class(0.4) == "score-mid"


def test_score_class_low() -> None:
    assert _score_class(0.3) == "score-low"
    assert _score_class(0.0) == "score-low"


def test_score_class_none() -> None:
    assert _score_class(None) == "score-na"


# ---------------------------------------------------------------------------
# _fmt_score
# ---------------------------------------------------------------------------

def test_fmt_score_none() -> None:
    assert _fmt_score(None) == "—"


def test_fmt_score_float() -> None:
    assert _fmt_score(0.75) == "0.75"
    assert _fmt_score(1.0) == "1.00"
    assert _fmt_score(0.0) == "0.00"


# ---------------------------------------------------------------------------
# _bar_color
# ---------------------------------------------------------------------------

def test_bar_color_high() -> None:
    assert _bar_color(0.8) == "#22c55e"
    assert _bar_color(0.7) == "#22c55e"


def test_bar_color_mid() -> None:
    assert _bar_color(0.5) == "#f59e0b"
    assert _bar_color(0.4) == "#f59e0b"


def test_bar_color_low() -> None:
    assert _bar_color(0.2) == "#ef4444"
    assert _bar_color(0.0) == "#ef4444"


# ---------------------------------------------------------------------------
# _category_badge и _pass_badge
# ---------------------------------------------------------------------------

def test_category_badge_positive() -> None:
    html = _category_badge("positive")
    assert 'class="badge badge-positive"' in html
    assert "positive" in html


def test_category_badge_negative() -> None:
    html = _category_badge("negative")
    assert "badge-negative" in html


def test_category_badge_multi_hop() -> None:
    html = _category_badge("multi_hop")
    assert "badge-multi_hop" in html
    assert "multi-hop" in html


def test_category_badge_unknown() -> None:
    html = _category_badge("custom_cat")
    assert "custom_cat" in html


def test_pass_badge_pass() -> None:
    html = _pass_badge(True)
    assert "PASS" in html
    assert "badge-pass" in html


def test_pass_badge_fail() -> None:
    html = _pass_badge(False)
    assert "FAIL" in html
    assert "badge-fail" in html


# ---------------------------------------------------------------------------
# _esc
# ---------------------------------------------------------------------------

def test_esc_ampersand() -> None:
    assert _esc("a & b") == "a &amp; b"


def test_esc_angle_brackets() -> None:
    assert _esc("<b>text</b>") == "&lt;b&gt;text&lt;/b&gt;"


def test_esc_quotes() -> None:
    assert _esc('"quoted"') == "&quot;quoted&quot;"


def test_esc_combined() -> None:
    result = _esc('<b>текст & "данные"</b>')
    assert "&lt;" in result
    assert "&gt;" in result
    assert "&amp;" in result
    assert "&quot;" in result


def test_esc_plain_text() -> None:
    assert _esc("plain text") == "plain text"


# ---------------------------------------------------------------------------
# Mock-компоненты
# ---------------------------------------------------------------------------

def test_mock_vector_db_search_returns_result() -> None:
    db = _MockVectorDB()
    result = db.search([0.1, 0.2, 0.3], n_results=3)
    assert len(result.ids) <= 3
    assert len(result.documents) == len(result.ids)
    assert len(result.distances) == len(result.ids)
    assert len(result.metadatas) == len(result.ids)


def test_mock_vector_db_search_respects_n_results() -> None:
    db = _MockVectorDB()
    result = db.search([], n_results=2)
    assert len(result.ids) <= 2


def test_mock_vector_db_count() -> None:
    db = _MockVectorDB()
    assert db.count() > 0


def test_mock_vector_db_add_and_delete_no_error() -> None:
    db = _MockVectorDB()
    db.add(ids=["x"], embeddings=[[0.1]])
    db.delete(ids=["x"])


@pytest.mark.asyncio
async def test_mock_llm_returns_string() -> None:
    llm = _MockLLMProvider()
    result = await llm.complete([])
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_mock_llm_model_attribute() -> None:
    llm = _MockLLMProvider()
    assert llm.model == "mock-llm-v1"


def test_build_mock_pipeline_creates_pipeline() -> None:
    from retrieval.pipeline import RAGPipeline
    pipeline = _build_mock_pipeline()
    assert isinstance(pipeline, RAGPipeline)


# ---------------------------------------------------------------------------
# generate_html_report
# ---------------------------------------------------------------------------

def _make_eval_result(n_positive: int = 3, n_negative: int = 2) -> dict:
    rows = [
        {"question": f"Q{i}", "answer": "Ответ", "contexts": ["ctx"],
         "ground_truth": "GT", "category": "positive"}
        for i in range(n_positive)
    ] + [
        {"question": f"NQ{i}", "answer": "Не нашёл информации.", "contexts": [],
         "ground_truth": "", "category": "negative"}
        for i in range(n_negative)
    ]
    return {
        "rows": rows,
        "ragas_df": None,
        "ragas_avg": {m: None for m in _METRIC_NAMES},
        "hallucination_results": [
            {"question": f"NQ{i}", "answer": "Не нашёл информации.", "passed": True}
            for i in range(n_negative)
        ],
    }


def test_generate_html_report_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    generate_html_report(
        result=_make_eval_result(),
        mode="mock",
        output_path=str(out),
        max_ragas_cases=5,
    )
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "<html" in content.lower()


def test_generate_html_report_contains_questions(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    generate_html_report(
        result=_make_eval_result(n_positive=2, n_negative=1),
        mode="mock",
        output_path=str(out),
        max_ragas_cases=5,
    )
    content = out.read_text(encoding="utf-8")
    assert "Q0" in content
    assert "NQ0" in content


def test_generate_html_report_shows_ragas_skip_message(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    generate_html_report(
        result=_make_eval_result(n_positive=2, n_negative=0),
        mode="mock",
        output_path=str(out),
        max_ragas_cases=5,
    )
    content = out.read_text(encoding="utf-8")
    assert "RAGAS" in content


def test_generate_html_report_shows_pass_fail_badges(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    result = _make_eval_result(n_negative=3)
    generate_html_report(result=result, mode="mock", output_path=str(out), max_ragas_cases=5)
    content = out.read_text(encoding="utf-8")
    assert "PASS" in content or "FAIL" in content


def test_generate_html_report_mode_shown_in_header(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    generate_html_report(
        result=_make_eval_result(),
        mode="ollama",
        output_path=str(out),
        max_ragas_cases=3,
    )
    content = out.read_text(encoding="utf-8")
    assert "ollama" in content


def test_generate_html_report_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "subdir" / "deep" / "report.html"
    generate_html_report(
        result=_make_eval_result(),
        mode="mock",
        output_path=str(out),
        max_ragas_cases=5,
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# ragas_eval.build_rows_from_testcases
# ---------------------------------------------------------------------------

_SAMPLE_TCS = [
    TestCase(question="Что такое Python?", ground_truth="ЯП", source_doc="doc.md", category="positive"),
    TestCase(question="Что такое Docker?", ground_truth="Контейнеры", source_doc="doc2.md", category="positive"),
]


def test_build_rows_preserves_question_count() -> None:
    rows = build_rows_from_testcases(_SAMPLE_TCS)
    assert len(rows) == len(_SAMPLE_TCS)


def test_build_rows_fields_present() -> None:
    rows = build_rows_from_testcases(_SAMPLE_TCS)
    for row in rows:
        assert "question" in row
        assert "ground_truth" in row
        assert "answer" in row
        assert "contexts" in row
        assert "source_doc" in row
        assert "category" in row


def test_build_rows_defaults_to_empty_answer_and_contexts() -> None:
    rows = build_rows_from_testcases(_SAMPLE_TCS)
    assert rows[0]["answer"] == ""
    assert rows[0]["contexts"] == []


def test_build_rows_uses_provided_answers() -> None:
    answers = {"Что такое Python?": "Язык программирования"}
    rows = build_rows_from_testcases(_SAMPLE_TCS, answer_by_question=answers)
    assert rows[0]["answer"] == "Язык программирования"
    assert rows[1]["answer"] == ""


def test_build_rows_uses_provided_contexts() -> None:
    ctx = {"Что такое Python?": ["контекст 1", "контекст 2"]}
    rows = build_rows_from_testcases(_SAMPLE_TCS, contexts_by_question=ctx)
    assert rows[0]["contexts"] == ["контекст 1", "контекст 2"]
    assert rows[1]["contexts"] == []


def test_build_rows_preserves_ground_truth() -> None:
    rows = build_rows_from_testcases(_SAMPLE_TCS)
    assert rows[0]["ground_truth"] == "ЯП"
    assert rows[1]["ground_truth"] == "Контейнеры"


# ---------------------------------------------------------------------------
# testcases_dataset
# ---------------------------------------------------------------------------

def test_get_testcases_total_count() -> None:
    cases = get_testcases()
    positives = sum(1 for c in cases if c.category == "positive")
    negatives = sum(1 for c in cases if c.category == "negative")
    multi_hops = sum(1 for c in cases if c.category == "multi_hop")
    assert len(cases) == positives + negatives + multi_hops


def test_get_testcases_category_counts() -> None:
    cases = get_testcases()
    assert sum(1 for c in cases if c.category == "negative") == 10
    assert sum(1 for c in cases if c.category == "multi_hop") == 5
    assert sum(1 for c in cases if c.category == "positive") >= 30


def test_get_testcases_positive_have_ground_truth() -> None:
    for tc in get_testcases():
        if tc.category == "positive":
            assert tc.question, f"positive case missing question: {tc}"
            assert tc.ground_truth, f"positive case missing ground_truth: {tc}"
            assert tc.source_doc, f"positive case missing source_doc: {tc}"


def test_get_testcases_negative_have_empty_fields() -> None:
    for tc in get_testcases():
        if tc.category == "negative":
            assert tc.ground_truth == ""
            assert tc.source_doc == ""


def test_get_testcases_valid_categories() -> None:
    valid = {"positive", "negative", "multi_hop"}
    for tc in get_testcases():
        assert tc.category in valid


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

def test_metric_names_contains_faithfulness() -> None:
    assert "faithfulness" in _METRIC_NAMES


def test_metric_names_contains_context_recall() -> None:
    assert "context_recall" in _METRIC_NAMES


def test_metric_names_has_four_metrics() -> None:
    assert len(_METRIC_NAMES) == 4


def test_pass_threshold_in_range() -> None:
    assert 0.0 < _PASS_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# _mock_embed_fn
# ---------------------------------------------------------------------------

def test_mock_embed_fn_returns_list() -> None:
    result = _mock_embed_fn("test text")
    assert isinstance(result, list)
    assert len(result) == 16
    assert all(v == 0.0 for v in result)


# ---------------------------------------------------------------------------
# generate_html_report с реальными RAGAS-баллами
# ---------------------------------------------------------------------------

def _make_eval_result_with_ragas(n_positive: int = 3) -> dict:
    rows = [
        {"question": f"Q{i}", "answer": "Ответ", "contexts": ["ctx"],
         "ground_truth": "GT", "category": "positive"}
        for i in range(n_positive)
    ]
    return {
        "rows": rows,
        "ragas_df": None,
        "ragas_avg": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.72,
            "context_precision": 0.45,
            "context_recall": 0.30,
        },
        "hallucination_results": [],
    }


def test_generate_html_report_with_ragas_scores_creates_bars(tmp_path: Path) -> None:
    out = tmp_path / "report_ragas.html"
    generate_html_report(
        result=_make_eval_result_with_ragas(),
        mode="ollama",
        output_path=str(out),
        max_ragas_cases=3,
    )
    content = out.read_text(encoding="utf-8")
    assert "metric-bar-fill" in content
    assert "faithfulness" in content
    assert "context_recall" in content


def test_generate_html_report_ragas_colors(tmp_path: Path) -> None:
    out = tmp_path / "report_colors.html"
    generate_html_report(
        result=_make_eval_result_with_ragas(),
        mode="ollama",
        output_path=str(out),
        max_ragas_cases=3,
    )
    content = out.read_text(encoding="utf-8")
    assert "#22c55e" in content   # green for score ≥ 0.7
    assert "#ef4444" in content   # red for score < 0.4


def test_generate_html_report_with_empty_ground_truth(tmp_path: Path) -> None:
    out = tmp_path / "report_empty_gt.html"
    result = {
        "rows": [
            {"question": "Q?", "answer": "A", "contexts": [],
             "ground_truth": "", "category": "negative"},
        ],
        "ragas_df": None,
        "ragas_avg": {m: None for m in _METRIC_NAMES},
        "hallucination_results": [
            {"question": "Q?", "answer": "A", "passed": False}
        ],
    }
    generate_html_report(result=result, mode="mock", output_path=str(out), max_ragas_cases=5)
    content = out.read_text(encoding="utf-8")
    assert "пусто" in content or "score-na" in content


# ---------------------------------------------------------------------------
# run_evaluation (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_evaluation_returns_expected_keys() -> None:
    pipeline = _build_mock_pipeline()
    tcs = [
        TestCase(question="Что такое PEP 8?", ground_truth="Стандарт", source_doc="", category="positive"),
        TestCase(question="Несуществующая GPU-функция?", ground_truth="", source_doc="", category="negative"),
    ]
    result = await run_evaluation(
        pipeline=pipeline,
        testcases=tcs,
        ragas_llm=None,
        ragas_embeddings=None,
        max_ragas_cases=1,
    )
    assert "rows" in result
    assert "ragas_df" in result
    assert "ragas_avg" in result
    assert "hallucination_results" in result


@pytest.mark.asyncio
async def test_run_evaluation_rows_count_matches_testcases() -> None:
    pipeline = _build_mock_pipeline()
    tcs = [
        TestCase(question=f"Вопрос {i}", ground_truth="GT", source_doc="", category="positive")
        for i in range(4)
    ]
    result = await run_evaluation(
        pipeline=pipeline,
        testcases=tcs,
        ragas_llm=None,
        max_ragas_cases=2,
    )
    assert len(result["rows"]) == 4


@pytest.mark.asyncio
async def test_run_evaluation_hallucination_only_for_negative() -> None:
    pipeline = _build_mock_pipeline()
    tcs = [
        TestCase(question="Что такое list?", ground_truth="Структура", source_doc="", category="positive"),
        TestCase(question="Несуществующая GPU?", ground_truth="", source_doc="", category="negative"),
        TestCase(question="Несуществующий PEP 999?", ground_truth="", source_doc="", category="negative"),
    ]
    result = await run_evaluation(pipeline=pipeline, testcases=tcs, max_ragas_cases=1)
    assert len(result["hallucination_results"]) == 2
    for hr in result["hallucination_results"]:
        assert "question" in hr
        assert "answer" in hr
        assert "passed" in hr


@pytest.mark.asyncio
async def test_run_evaluation_ragas_avg_none_in_mock_mode() -> None:
    pipeline = _build_mock_pipeline()
    tcs = [
        TestCase(question="Что такое декоратор?", ground_truth="Функция", source_doc="", category="positive"),
    ]
    result = await run_evaluation(pipeline=pipeline, testcases=tcs, ragas_llm=None, max_ragas_cases=5)
    assert result["ragas_df"] is None
    assert all(v is None for v in result["ragas_avg"].values())


@pytest.mark.asyncio
async def test_run_evaluation_exception_in_pipeline_yields_empty_answer() -> None:
    class _BrokenPipeline:
        async def run_detailed(self, query: str) -> tuple[str, list[str]]:
            raise RuntimeError("intentional test error")

    tcs = [
        TestCase(question="Вопрос", ground_truth="GT", source_doc="", category="positive"),
    ]
    result = await run_evaluation(pipeline=_BrokenPipeline(), testcases=tcs, max_ragas_cases=1)
    assert result["rows"][0]["answer"] == ""
    assert result["rows"][0]["contexts"] == []


# ---------------------------------------------------------------------------
# generate_html_report с частичными RAGAS-баллами (None + non-None)
# ---------------------------------------------------------------------------

def test_generate_html_report_partial_ragas_avg(tmp_path: Path) -> None:
    out = tmp_path / "report_partial.html"
    result = {
        "rows": [{"question": "Q", "answer": "A", "contexts": [], "ground_truth": "GT", "category": "positive"}],
        "ragas_df": None,
        "ragas_avg": {
            "faithfulness": 0.80,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": 0.65,
        },
        "hallucination_results": [],
    }
    generate_html_report(result=result, mode="mock", output_path=str(out), max_ragas_cases=5)
    content = out.read_text(encoding="utf-8")
    assert "faithfulness" in content
    assert "context_recall" in content


# ---------------------------------------------------------------------------
# main() CLI — выполняется в mock-режиме с перехватом os._exit
# ---------------------------------------------------------------------------

import sys as _sys
import os as _os


def test_main_mock_mode_creates_report(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "eval_report.html"
    monkeypatch.setattr(_sys, "argv", ["eval_runner", "--output", str(out)])

    exit_calls: list[int] = []
    monkeypatch.setattr(_os, "_exit", lambda code: exit_calls.append(code))

    from evaluation.eval_runner import main
    main()

    assert exit_calls == [0]
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_main_mock_mode_quiet_flag(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "quiet_report.html"
    monkeypatch.setattr(_sys, "argv", ["eval_runner", "--output", str(out), "--max-cases", "2"])

    monkeypatch.setattr(_os, "_exit", lambda code: None)

    from evaluation.eval_runner import main
    main()
    assert out.exists()
