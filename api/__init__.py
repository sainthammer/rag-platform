"""Пакет api: FastAPI-приложение RAG-платформы.

Экспортирует:
    app — ASGI-приложение FastAPI с маршрутами /health и /v1/metrics.

Запуск::

    uvicorn api.app:app --host 0.0.0.0 --port 8080
"""

from api.app import app

__all__ = ["app"]
