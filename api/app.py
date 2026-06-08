"""FastAPI-приложение RAG-платформы.

Точка входа ASGI-сервера. Содержит базовые маршруты и монтирует
Prometheus-метрики на ``GET /v1/metrics``.

Маршруты:
    GET /health       — проверка работоспособности сервера (liveness probe)
    GET /v1/metrics   — метрики в формате Prometheus exposition format

Запуск через uvicorn::

    uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload

После старта:
    http://localhost:8080/health     → {"status": "ok"}
    http://localhost:8080/v1/metrics → текст с метриками Prometheus
"""

from fastapi import FastAPI

from observability.metrics import make_metrics_app
from observability.tracing import instrument_fastapi

app = FastAPI(
    title="RAG Platform",
    version="0.1.0",
    description="RAG-платформа с поддержкой MCP, observability и evaluation.",
)


@app.get("/health", tags=["infra"])
async def health() -> dict[str, str]:
    """Проверить работоспособность сервера.

    Используется как liveness probe в Kubernetes / Docker Compose.
    Всегда возвращает 200 OK, если сервер запущен.
    """
    return {"status": "ok"}


# Монтируем ASGI-приложение Prometheus.
# make_asgi_app() экспортирует все метрики из глобального REGISTRY prometheus_client,
# включая метрики из observability/metrics.py, которые обновляются в pipeline и сервисах.
# Путь /v1/metrics — стандартный для Prometheus scraping через Kubernetes ServiceMonitor.
app.mount("/v1/metrics", make_metrics_app())

# Инструментируем FastAPI для автоматических OTel-спанов на каждый HTTP-запрос.
# Вызывать ПОСЛЕ создания app и регистрации маршрутов.
instrument_fastapi(app)
