# Модуль `observability/` — трейсинг OpenTelemetry и Prometheus-метрики

Модуль отвечает за distributed tracing (OpenTelemetry) и Prometheus-метрики.

- **Трейсинг**: каждый RAG-запрос создаёт иерархию спанов `rag.query → rag.retrieve → rag.generate`.
  Спаны содержат атрибуты `query`, `model`, `chunk_count`, `latency_ms`.
- **Метрики**: Counter и Histogram для всех ключевых операций (RAG, Embedding, LLM, Retrieval).
  Экспортируются через эндпоинт `GET /v1/metrics` в формате Prometheus exposition.

---

## Структура файлов

```
observability/
├── tracing.py   # setup_tracing + instrument_fastapi
├── metrics.py   # Counter, Histogram для всех операций + make_metrics_app()
├── __init__.py  # re-export публичного API модуля
└── README.md
```

---

## Публичное API модуля

```python
from observability import setup_tracing, instrument_fastapi, make_metrics_app
from observability.metrics import (
    RAG_REQUESTS_TOTAL, RAG_LATENCY_SECONDS,
    EMBEDDING_REQUESTS_TOTAL, EMBEDDING_LATENCY_SECONDS,
    LLM_REQUESTS_TOTAL, LLM_LATENCY_SECONDS,
    RETRIEVAL_CHUNK_COUNT,
)
```

---

## `setup_tracing()` — настройка TracerProvider и OTLP exporter

Функция `setup_tracing(service_name="rag-platform")`:

- создаёт `TracerProvider` и регистрирует его глобально
- настраивает `BatchSpanProcessor` (отправка спанов пачками)
- настраивает `OTLPSpanExporter` (gRPC)

Endpoint для экспорта берётся из `config.py`:

- `OTEL_EXPORTER_OTLP_ENDPOINT` → `settings.otel_exporter_otlp_endpoint`

Пример значения:

- `http://localhost:4317`

> Это endpoint приёмника телеметрии, а не endpoint вашего API.

---

## `instrument_fastapi()` — автоинструментация FastAPI

Функция `instrument_fastapi(app)` подключает `FastAPIInstrumentor`.

Что даёт:

- автоматические серверные спаны на каждый HTTP-запрос
- атрибуты уровня route/status_code и т.п.

Эту функцию нужно вызвать **после** создания `FastAPI()`.

---

## Зависимости

Для работы трейсинга нужны зависимости OpenTelemetry SDK и OTLP exporter.
Для автоинструментации FastAPI отдельно нужен пакет:

- `opentelemetry-instrumentation-fastapi`

Если пакет не установлен, `instrument_fastapi()` выбросит `ImportError` с подсказкой.

---

## Запуск `example.py`

### Режим по умолчанию — консоль (без внешних сервисов)

```powershell
# из корня проекта
python observability\example.py
```

Демонстрирует:
- Span-ы в stdout через `ConsoleSpanExporter`
- Prometheus-метрики: имитация запросов + вывод `generate_latest()`

### Режим server — FastAPI с /v1/metrics

```powershell
python observability\example.py server
```

Запускает uvicorn на `http://localhost:8080`:
- `GET /health` → `{"status": "ok"}`
- `GET /v1/metrics` → текст метрик Prometheus

Проверить:

```powershell
curl http://localhost:8080/v1/metrics
# или в браузере: http://localhost:8080/v1/metrics
```

### Режим Jaeger — отправка span-ов в реальный коллектор

**1. Запустить Jaeger через Docker:**

```powershell
docker run --rm --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
```

**2. Задать endpoint в `.env`:**

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

**3. Запустить example в jaeger-режиме:**

```powershell
python observability\example.py jaeger
```

**4. Открыть UI:** `http://localhost:16686` → выбрать сервис `rag-platform-example` → `Find Traces`.

---

## Prometheus-метрики (`observability/metrics.py`)

### Список метрик

| Метрика | Тип | Labels | Описание |
|---|---|---|---|
| `rag_requests_total` | Counter | `status` | Число RAG-запросов |
| `rag_latency_seconds` | Histogram | — | Полная задержка RAG-цикла |
| `embedding_requests_total` | Counter | `model`, `status` | Число embedding-запросов |
| `embedding_latency_seconds` | Histogram | `model` | Задержка получения эмбеддинга |
| `llm_requests_total` | Counter | `provider`, `model`, `status` | Число LLM-запросов |
| `llm_latency_seconds` | Histogram | `provider`, `model` | Задержка ответа LLM |
| `retrieval_chunk_count` | Histogram | — | Число чанков после retrieval |

### Использование в коде

```python
from observability.metrics import RAG_REQUESTS_TOTAL, RAG_LATENCY_SECONDS

RAG_REQUESTS_TOTAL.labels(status="success").inc()
RAG_LATENCY_SECONDS.observe(1.23)
```

Метрики обновляются автоматически в `RAGPipeline.run()` и `EmbeddingService.embed()`.

---

## GET /v1/metrics — эндпоинт Prometheus

### Запуск API-сервера

```powershell
# из корня проекта
uvicorn api.app:app --host 0.0.0.0 --port 8080
```

### Проверка метрик

```powershell
# в браузере или curl:
curl http://localhost:8080/v1/metrics
```

Ответ — текст в формате Prometheus exposition format:

```
# HELP rag_requests_total Общее число запросов к RAG-пайплайну
# TYPE rag_requests_total counter
rag_requests_total{status="success"} 3.0
rag_requests_total{status="error"} 1.0
# HELP rag_latency_seconds Задержка полного RAG-цикла ...
...
```

### Подключение Prometheus (prometheus.yml)

```yaml
scrape_configs:
  - job_name: rag-platform
    static_configs:
      - targets: ["localhost:8080"]
    metrics_path: /v1/metrics
```

---

## Запуск `example.py`

### Режим по умолчанию — консольный вывод спанов + метрики
