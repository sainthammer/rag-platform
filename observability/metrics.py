"""Prometheus-метрики платформы RAG.

Все метрики регистрируются в глобальном реестре prometheus_client.
Импортируйте нужные объекты напрямую и вызывайте их методы там, где
происходит измеряемое событие.

Экспортирует:
    RAG_REQUESTS_TOTAL        — Counter: число запросов к RAG-пайплайну (по статусу)
    RAG_LATENCY_SECONDS       — Histogram: задержка полного RAG-цикла
    EMBEDDING_REQUESTS_TOTAL  — Counter: число запросов к EmbeddingService (модель, статус)
    EMBEDDING_LATENCY_SECONDS — Histogram: задержка получения эмбеддинга (по модели)
    LLM_REQUESTS_TOTAL        — Counter: число запросов к LLM (провайдер, модель, статус)
    LLM_LATENCY_SECONDS       — Histogram: задержка ответа LLM (провайдер, модель)
    RETRIEVAL_CHUNK_COUNT     — Histogram: число чанков, отобранных при retrieval
    make_metrics_app          — фабрика ASGI-приложения для монтирования в FastAPI
"""

from prometheus_client import Counter, Histogram, make_asgi_app

# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Общее число запросов к RAG-пайплайну",
    labelnames=["status"],  # success | error
)

RAG_LATENCY_SECONDS = Histogram(
    "rag_latency_seconds",
    "Задержка полного RAG-цикла (retrieval + generation), секунды",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# ---------------------------------------------------------------------------
# Embedding Service
# ---------------------------------------------------------------------------

EMBEDDING_REQUESTS_TOTAL = Counter(
    "embedding_requests_total",
    "Число запросов к EmbeddingService",
    labelnames=["model", "status"],  # status: success | error
)

EMBEDDING_LATENCY_SECONDS = Histogram(
    "embedding_latency_seconds",
    "Задержка получения эмбеддинга, секунды",
    labelnames=["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Число запросов к LLM-провайдеру",
    labelnames=["provider", "model", "status"],  # status: success | error
)

LLM_LATENCY_SECONDS = Histogram(
    "llm_latency_seconds",
    "Задержка ответа LLM, секунды",
    labelnames=["provider", "model"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_CHUNK_COUNT = Histogram(
    "retrieval_chunk_count",
    "Число чанков, отобранных после token-budget фильтрации",
    buckets=[0, 1, 2, 3, 5, 8, 13, 21],
)


def make_metrics_app():
    """Вернуть ASGI-приложение Prometheus для монтирования в FastAPI.

    Возвращаемое приложение отвечает на GET-запросы в формате
    Prometheus exposition format (text/plain; version=0.0.4).

    Пример монтирования::

        from fastapi import FastAPI
        from observability.metrics import make_metrics_app

        app = FastAPI()
        app.mount("/v1/metrics", make_metrics_app())

    После старта сервера метрики доступны по адресу http://localhost:8080/v1/metrics.
    """
    return make_asgi_app()
