# Модуль 9 — observability/

**Файлы:** `observability/metrics.py`, `observability/tracing.py`

Два инструмента мониторинга: метрики через Prometheus и трейсинг через OpenTelemetry.

## Зачем нужна наблюдаемость

В production нужно знать:
- Сколько запросов обрабатывается в секунду?
- Какая задержка у RAG-цикла в p95?
- Где тормозит — на embedding, поиске или LLM?
- Сколько ошибок?

Без инструментации — это «чёрный ящик». С инструментацией — дашборды и алерты.

## observability/metrics.py — Prometheus

### Типы метрик

**Counter** — монотонно растёт, только увеличивается:
```python
RAG_REQUESTS_TOTAL = Counter("rag_requests_total", "...", labelnames=["status"])
RAG_REQUESTS_TOTAL.labels(status="success").inc()  # +1
```

**Histogram** — распределение значений по бакетам:
```python
RAG_LATENCY_SECONDS = Histogram("rag_latency_seconds", "...",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0])
RAG_LATENCY_SECONDS.observe(0.85)  # записываем значение
```

### Все метрики проекта

| Метрика | Тип | Labels | Что измеряет |
|---|---|---|---|
| `rag_requests_total` | Counter | `status` | Число запросов к RAG |
| `rag_latency_seconds` | Histogram | — | Задержка полного RAG-цикла |
| `embedding_requests_total` | Counter | `model`, `status` | Запросы к EmbeddingService |
| `embedding_latency_seconds` | Histogram | `model` | Задержка embedding |
| `llm_requests_total` | Counter | `provider`, `model`, `status` | Запросы к LLM |
| `llm_latency_seconds` | Histogram | `provider`, `model` | Задержка LLM |
| `retrieval_chunk_count` | Histogram | — | Число чанков после budget-фильтрации |

### Как Prometheus собирает метрики

Prometheus работает по модели **pull**: он сам приходит к приложению и "скрейпит" метрики с эндпоинта:

```
Prometheus ──GET /v1/metrics──▶ FastAPI
           ◀──text/plain──────── метрики в Exposition Format
```

Пример экспортируемого текста:
```
rag_requests_total{status="success"} 42
rag_requests_total{status="error"} 3
rag_latency_seconds_bucket{le="0.5"} 38
rag_latency_seconds_bucket{le="1.0"} 41
```

`make_metrics_app()` — фабрика ASGI-приложения. В `app.py` монтируется на `/v1/metrics`.

## observability/tracing.py — OpenTelemetry

### Что такое трейс

Трейс — запись "пути" одного запроса через систему. Состоит из **span**'ов (промежутков):

```
HTTP POST /v1/ask                     [span: 850ms]
  ├── embed(question)                  [span: 50ms]
  ├── db.search()                      [span: 20ms]
  └── llm.complete()                   [span: 780ms]
```

Каждый span знает: когда начался, когда закончился, какие атрибуты (model, status...).

### Настройка

```python
def setup_tracing(service_name="rag-platform") -> TracerProvider:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,  # http://jaeger:4317
        insecure=True
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider
```

Экспорт через **OTLP gRPC** в Jaeger (или любой другой OTLP-совместимый коллектор: Tempo, Zipkin...).

### Инструментация FastAPI

```python
instrument_fastapi(app)
# FastAPIInstrumentor автоматически создаёт span для каждого HTTP-запроса
```

После этого каждый запрос к API автоматически записывается как трейс — без ручной инструментации эндпоинтов.

### Интеграция с X-Request-ID

`RequestIDMiddleware` прокидывает `request_id` в текущий OTel-span:

```python
span = trace.get_current_span()
if span.is_recording():
    span.set_attribute("http.request_id", request_id)
```

Это связывает HTTP-заголовок с трейсом: найдя `X-Request-ID` в логах, можно открыть этот трейс в Jaeger.

## Grafana дашборды

В `observability/grafana/provisioning/` лежат заготовленные дашборды:
- Запросы в секунду (RPS) по статусам
- Latency p50/p95/p99 для RAG, embedding и LLM
- Процент ошибок
- Число чанков на запрос

## Схема взаимодействия сервисов

```
FastAPI app
  │
  ├── Prometheus метрики ──── pull ──── Prometheus
  │                                         │
  │                                         ▼
  │                                      Grafana
  │                                    (дашборды)
  │
  └── OTel tracing ──── OTLP gRPC ──── Jaeger
                         port 4317     (UI: 16686)
```

## Как добавить метрику в новый код

```python
from observability.metrics import LLM_REQUESTS_TOTAL, LLM_LATENCY_SECONDS
import time

start = time.monotonic()
try:
    result = await llm.complete(messages)
    LLM_REQUESTS_TOTAL.labels(provider="openai", model="gpt-4o", status="success").inc()
except Exception:
    LLM_REQUESTS_TOTAL.labels(provider="openai", model="gpt-4o", status="error").inc()
    raise
finally:
    LLM_LATENCY_SECONDS.labels(provider="openai", model="gpt-4o").observe(time.monotonic() - start)
```
