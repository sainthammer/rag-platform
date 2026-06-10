"""Unit-тесты для модуля observability.

Покрывают:
  - setup_tracing(): создание TracerProvider, возврат корректного типа
  - make_metrics_app(): возврат ASGI-приложения
  - Prometheus-счётчики и гистограммы: наличие и корректность работы
"""

from __future__ import annotations

import pytest

from observability.metrics import (
    EMBEDDING_LATENCY_SECONDS,
    EMBEDDING_REQUESTS_TOTAL,
    LLM_LATENCY_SECONDS,
    LLM_REQUESTS_TOTAL,
    RAG_LATENCY_SECONDS,
    RAG_REQUESTS_TOTAL,
    RETRIEVAL_CHUNK_COUNT,
    make_metrics_app,
)


# ---------------------------------------------------------------------------
# make_metrics_app
# ---------------------------------------------------------------------------

def test_make_metrics_app_returns_callable() -> None:
    app = make_metrics_app()
    assert callable(app)


def test_make_metrics_app_returns_new_instance_each_call() -> None:
    app1 = make_metrics_app()
    app2 = make_metrics_app()
    # ASGI-приложения — отдельные объекты
    assert app1 is not app2


# ---------------------------------------------------------------------------
# Prometheus-метрики: наличие и базовые операции
# ---------------------------------------------------------------------------

def test_rag_requests_total_is_counter() -> None:
    from prometheus_client import Counter
    assert isinstance(RAG_REQUESTS_TOTAL, Counter)


def test_rag_latency_seconds_is_histogram() -> None:
    from prometheus_client import Histogram
    assert isinstance(RAG_LATENCY_SECONDS, Histogram)


def test_embedding_requests_total_is_counter() -> None:
    from prometheus_client import Counter
    assert isinstance(EMBEDDING_REQUESTS_TOTAL, Counter)


def test_embedding_latency_seconds_is_histogram() -> None:
    from prometheus_client import Histogram
    assert isinstance(EMBEDDING_LATENCY_SECONDS, Histogram)


def test_llm_requests_total_is_counter() -> None:
    from prometheus_client import Counter
    assert isinstance(LLM_REQUESTS_TOTAL, Counter)


def test_llm_latency_seconds_is_histogram() -> None:
    from prometheus_client import Histogram
    assert isinstance(LLM_LATENCY_SECONDS, Histogram)


def test_retrieval_chunk_count_is_histogram() -> None:
    from prometheus_client import Histogram
    assert isinstance(RETRIEVAL_CHUNK_COUNT, Histogram)


def test_rag_requests_total_increment() -> None:
    RAG_REQUESTS_TOTAL.labels(status="success").inc()


def test_rag_latency_seconds_observe() -> None:
    RAG_LATENCY_SECONDS.observe(0.5)


def test_embedding_requests_total_increment() -> None:
    EMBEDDING_REQUESTS_TOTAL.labels(model="fake", status="success").inc()


def test_embedding_latency_seconds_observe() -> None:
    EMBEDDING_LATENCY_SECONDS.labels(model="fake").observe(0.01)


def test_llm_requests_total_increment() -> None:
    LLM_REQUESTS_TOTAL.labels(provider="mock", model="mock-llm-v1", status="success").inc()


def test_llm_latency_seconds_observe() -> None:
    LLM_LATENCY_SECONDS.labels(provider="mock", model="mock-llm-v1").observe(1.5)


def test_retrieval_chunk_count_observe() -> None:
    RETRIEVAL_CHUNK_COUNT.observe(3)


# ---------------------------------------------------------------------------
# setup_tracing
# ---------------------------------------------------------------------------

def test_setup_tracing_returns_tracer_provider() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from observability.tracing import setup_tracing

    provider = setup_tracing("test-service")
    try:
        assert isinstance(provider, TracerProvider)
    finally:
        provider.shutdown()


def test_setup_tracing_sets_global_provider() -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from observability.tracing import setup_tracing

    provider = setup_tracing("test-service-2")
    try:
        global_provider = trace.get_tracer_provider()
        assert isinstance(global_provider, TracerProvider)
    finally:
        provider.shutdown()


def test_setup_tracing_custom_service_name() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from observability.tracing import setup_tracing

    provider = setup_tracing("my-custom-service")
    try:
        assert isinstance(provider, TracerProvider)
    finally:
        provider.shutdown()
