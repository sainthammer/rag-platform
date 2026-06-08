"""Демонстрация модуля observability: трейсинг OpenTelemetry и Prometheus-метрики.

Запуск:
    PYTHONPATH=. .venv/Scripts/python observability/example.py           # вывод в консоль
    PYTHONPATH=. .venv/Scripts/python observability/example.py jaeger    # отправка в Jaeger
    PYTHONPATH=. .venv/Scripts/python observability/example.py server    # запуск API с /v1/metrics

Режим jaeger требует запущенного Jaeger all-in-one и переменной OTEL_EXPORTER_OTLP_ENDPOINT.
Поднять локально через Docker:
    docker run --rm --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
Затем открыть UI: http://localhost:16686

Что демонстрируется:
    1. Инициализация TracerProvider (ConsoleSpanExporter или OTLP → Jaeger).
    2. Ручное создание корневого и вложенного span с атрибутами.
    3. Инструментация FastAPI-приложения через instrument_fastapi().
    4. Prometheus-метрики: Counter, Histogram, generate_latest().
    5. (server-режим) Запуск FastAPI-сервера с /v1/metrics.
"""

import sys

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter


# ---------------------------------------------------------------------------
# Вспомогательная функция: TracerProvider с выводом в консоль
# ---------------------------------------------------------------------------

def _setup_console_tracing(service_name: str = "rag-platform-example") -> TracerProvider:
    """Создаёт TracerProvider с ConsoleSpanExporter — не требует OTLP-сервера."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


# ---------------------------------------------------------------------------
# 1. Ручные span-ы с атрибутами
# ---------------------------------------------------------------------------

def demo_spans() -> None:
    """Создать корневой span и вложенный дочерний span с атрибутами."""
    tracer = trace.get_tracer("rag-platform")

    print("─── 1. Ручные spans ───")
    with tracer.start_as_current_span("rag.query") as root:
        root.set_attribute("query", "Что такое RAG?")
        root.set_attribute("n_results", 5)

        with tracer.start_as_current_span("rag.retrieve") as child:
            child.set_attribute("vector_db", "qdrant")
            child.set_attribute("chunks_found", 3)

        with tracer.start_as_current_span("rag.generate") as child:
            child.set_attribute("llm_provider", "openai")
            child.set_attribute("model", "gpt-4o-mini")

    print("  spans 'rag.query', 'rag.retrieve', 'rag.generate' записаны выше ↑\n")


# ---------------------------------------------------------------------------
# 2. instrument_fastapi()
# ---------------------------------------------------------------------------

def demo_fastapi_instrumentation() -> None:
    """Инструментировать FastAPI-приложение через observability.instrument_fastapi()."""
    try:
        from fastapi import FastAPI
        from observability.tracing import instrument_fastapi
    except ImportError as exc:
        print(f"  Пропущено (отсутствует зависимость): {exc}\n")
        return

    print("─── 2. Инструментация FastAPI ───")
    app = FastAPI(title="RAG Platform")
    instrument_fastapi(app)
    routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
    print(f"  Приложение '{app.title}' инструментировано.")
    print(f"  Маршруты: {routes}")
    print("  Все входящие HTTP-запросы автоматически создадут span в трейсе.\n")


# ---------------------------------------------------------------------------
# 3. Prometheus-метрики
# ---------------------------------------------------------------------------

def demo_metrics() -> None:
    """Показать, как работают Counter и Histogram из observability/metrics.py.

    Имитирует несколько RAG-запросов: обновляет счётчики и гистограммы,
    затем печатает актуальный срез метрик через generate_latest().
    """
    from prometheus_client import generate_latest

    from observability.metrics import (
        EMBEDDING_LATENCY_SECONDS,
        EMBEDDING_REQUESTS_TOTAL,
        LLM_LATENCY_SECONDS,
        LLM_REQUESTS_TOTAL,
        RAG_LATENCY_SECONDS,
        RAG_REQUESTS_TOTAL,
        RETRIEVAL_CHUNK_COUNT,
    )

    print("─── 3. Prometheus-метрики ───")

    # Имитация трёх успешных RAG-запросов с разной задержкой
    for latency in (0.4, 0.9, 2.1):
        RAG_REQUESTS_TOTAL.labels(status="success").inc()
        RAG_LATENCY_SECONDS.observe(latency)

    # Имитация одного ошибочного RAG-запроса
    RAG_REQUESTS_TOTAL.labels(status="error").inc()

    # Имитация retrieval: 3 и 5 чанков в двух запросах
    RETRIEVAL_CHUNK_COUNT.observe(3)
    RETRIEVAL_CHUNK_COUNT.observe(5)

    # Имитация embedding-запросов
    EMBEDDING_REQUESTS_TOTAL.labels(model="text-embedding-3-small", status="success").inc()
    EMBEDDING_LATENCY_SECONDS.labels(model="text-embedding-3-small").observe(0.07)

    # Имитация LLM-запросов
    LLM_REQUESTS_TOTAL.labels(provider="openai", model="gpt-4o-mini", status="success").inc()
    LLM_LATENCY_SECONDS.labels(provider="openai", model="gpt-4o-mini").observe(1.2)

    # Генерируем текст метрик в Prometheus exposition format
    metrics_text = generate_latest().decode("utf-8")

    # Печатаем только строки наших метрик (отфильтровываем стандартные python_* / process_*)
    our_prefixes = ("rag_", "embedding_", "llm_", "retrieval_")
    lines = [
        line for line in metrics_text.splitlines()
        if (
            any(line.startswith(pfx) for pfx in our_prefixes)
            or (line.startswith("# ") and any(pfx in line for pfx in our_prefixes))
        )
    ]

    print("  Срез метрик (формат Prometheus exposition):")
    for line in lines:
        print(f"    {line}")
    print()
    print("  В production этот текст отдаёт GET /v1/metrics.")
    print("  Prometheus scrape-job забирает его каждые N секунд.\n")


# ---------------------------------------------------------------------------
# 4. Режим server: FastAPI с /v1/metrics
# ---------------------------------------------------------------------------

def demo_server() -> None:
    """Запустить FastAPI-сервер с эндпоинтом /v1/metrics.

    После старта:
        GET http://localhost:8080/health      → {"status": "ok"}
        GET http://localhost:8080/v1/metrics  → метрики Prometheus

    Остановить: Ctrl+C.
    """
    try:
        import uvicorn
    except ImportError:
        print("  Пропущено: установите uvicorn — pip install uvicorn[standard]")
        return

    from api.app import app

    print("─── server-режим ───")
    print("  Запускаем uvicorn на http://localhost:8080")
    print("  GET /v1/health    → {\"status\": \"ok\", ...}")
    print("  GET /v1/metrics   → метрики Prometheus")
    print("  Остановить: Ctrl+C\n")

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


# ---------------------------------------------------------------------------
# 5. Режим Jaeger: реальный setup_tracing() с OTLP-экспортом
# ---------------------------------------------------------------------------

def demo_jaeger() -> None:
    """Отправить span-ы в Jaeger через setup_tracing() (реальный OTLP gRPC).

    Jaeger должен быть запущен и принимать OTLP на порту 4317.
    OTEL_EXPORTER_OTLP_ENDPOINT должен быть задан в .env (или окружении).
    """
    import os
    from observability.tracing import setup_tracing

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    print(f"─── Jaeger mode | endpoint={endpoint} ───")

    provider = setup_tracing(service_name="rag-platform-example")
    tracer = trace.get_tracer("rag-platform")

    with tracer.start_as_current_span("rag.query") as root:
        root.set_attribute("query", "Что такое RAG?")
        root.set_attribute("n_results", 5)

        with tracer.start_as_current_span("rag.retrieve") as child:
            child.set_attribute("vector_db", "qdrant")
            child.set_attribute("chunks_found", 3)

        with tracer.start_as_current_span("rag.generate") as child:
            child.set_attribute("llm_provider", "openai")
            child.set_attribute("model", "gpt-4o-mini")

    provider.shutdown()
    print(f"  Spans отправлены в Jaeger. Откройте http://localhost:16686")
    print(f"  Выберите сервис 'rag-platform-example' и нажмите 'Find Traces'.\n")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа демонстрации observability-модуля.

    Без аргументов: консольный вывод span-ов + демонстрация метрик.
    jaeger:  отправка span-ов в Jaeger через OTLP.
    server:  запуск FastAPI с /health и /v1/metrics.
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "console"

    if mode == "jaeger":
        demo_jaeger()
    elif mode == "server":
        demo_server()
    else:
        provider = _setup_console_tracing()
        demo_spans()
        demo_fastapi_instrumentation()
        provider.shutdown()
        demo_metrics()

    print("✓ Готово")


if __name__ == "__main__":
    main()
