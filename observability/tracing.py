from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI


def setup_tracing(service_name: str = "rag-platform") -> TracerProvider:
    """Настраивает OpenTelemetry tracing и OTLP экспорт.

    Экспорт выполняется через OTLP endpoint (gRPC), который задаётся переменной окружения
    `OTEL_EXPORTER_OTLP_ENDPOINT` (см. `config.py`). Этот endpoint можно направить на Jaeger,
    если Jaeger принимает OTLP (например, через jaeger-collector OTLP gRPC).

    Возвращает установленный `TracerProvider`.
    """

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider


def instrument_fastapi(app: "FastAPI") -> None:
    """Инструментирует FastAPI-приложение через `FastAPIInstrumentor`.

    Вызывать после создания `FastAPI()`.

    Примечание: если пакет `opentelemetry-instrumentation-fastapi` не установлен,
    будет выброшен `ImportError` с подсказкой.
    """

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Отсутствует зависимость 'opentelemetry-instrumentation-fastapi'. "
            "Установите зависимости проекта."
        ) from e

    FastAPIInstrumentor.instrument_app(app)
