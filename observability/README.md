# Модуль `observability/` — трейсинг OpenTelemetry

Модуль отвечает за базовую настройку distributed tracing через OpenTelemetry (OTel).

Текущая реализация фокусируется на trace'ах (спанах): приложение создаёт спаны, а затем
OTLP exporter отправляет их на внешний приёмник (Jaeger Collector / OpenTelemetry Collector).

---

## Структура файлов

```
observability/
├── tracing.py   # setup_tracing + instrument_fastapi
├── __init__.py  # re-export публичного API модуля
└── README.md
```

---

## Публичное API модуля

```python
from observability import setup_tracing, instrument_fastapi
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
