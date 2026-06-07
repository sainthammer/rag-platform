"""Демонстрация модуля observability: трейсинг OpenTelemetry.

Запуск:
    PYTHONPATH=. .venv/Scripts/python observability/example.py           # вывод в консоль
    PYTHONPATH=. .venv/Scripts/python observability/example.py jaeger    # отправка в Jaeger

Режим jaeger требует запущенного Jaeger all-in-one и переменной OTEL_EXPORTER_OTLP_ENDPOINT.
Поднять локально через Docker:
    docker run --rm --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
Затем открыть UI: http://localhost:16686

Что демонстрируется:
    1. Инициализация TracerProvider (ConsoleSpanExporter или OTLP → Jaeger).
    2. Ручное создание корневого и вложенного span с атрибутами.
    3. Инструментация FastAPI-приложения через instrument_fastapi().
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
# 3. Режим Jaeger: реальный setup_tracing() с OTLP-экспортом
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
    use_jaeger = len(sys.argv) > 1 and sys.argv[1] == "jaeger"

    if use_jaeger:
        demo_jaeger()
    else:
        provider = _setup_console_tracing()
        demo_spans()
        demo_fastapi_instrumentation()
        provider.shutdown()

    print("✓ Готово")


if __name__ == "__main__":
    main()
