"""Пакет observability: трейсинг на базе OpenTelemetry.

Экспортирует:
    setup_tracing       — инициализация TracerProvider с OTLP gRPC-экспортом.
    instrument_fastapi  — автоматическая инструментация FastAPI-приложения.
"""

from observability.tracing import instrument_fastapi, setup_tracing

__all__ = [
    "setup_tracing",
    "instrument_fastapi",
]

