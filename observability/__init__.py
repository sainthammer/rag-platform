"""Пакет observability: трейсинг OpenTelemetry и Prometheus-метрики.

Экспортирует:
    setup_tracing       — инициализация TracerProvider с OTLP gRPC-экспортом.
    instrument_fastapi  — автоматическая инструментация FastAPI-приложения.
    make_metrics_app    — ASGI-приложение для монтирования эндпоинта /v1/metrics.

Prometheus-метрики (импортируйте напрямую из observability.metrics):
    RAG_REQUESTS_TOTAL, RAG_LATENCY_SECONDS
    EMBEDDING_REQUESTS_TOTAL, EMBEDDING_LATENCY_SECONDS
    LLM_REQUESTS_TOTAL, LLM_LATENCY_SECONDS
    RETRIEVAL_CHUNK_COUNT
"""

from observability.metrics import make_metrics_app
from observability.tracing import instrument_fastapi, setup_tracing

__all__ = [
    "setup_tracing",
    "instrument_fastapi",
    "make_metrics_app",
]

