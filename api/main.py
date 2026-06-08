"""Точка входа ASGI-приложения для uvicorn / docker-compose.

Этот модуль служит единственной точкой входа, чтобы команда запуска
оставалась стабильной вне зависимости от внутренней структуры пакета api/.

Запуск::

    uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
"""

from api.app import app  # noqa: F401  # re-export для uvicorn

__all__ = ["app"]
