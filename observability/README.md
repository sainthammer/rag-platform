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

---

## Запуск `example.py`

### Режим по умолчанию — вывод span-ов в консоль (без внешних сервисов)

```powershell
# из корня проекта
python observability\example.py
```

Span-ы печатаются в stdout в виде JSON через `ConsoleSpanExporter`.
Полезно для отладки трейсинга без Jaeger.

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
