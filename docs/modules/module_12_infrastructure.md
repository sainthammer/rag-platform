# Модуль 12 — Инфраструктура

**Файлы:** `Dockerfile`, `docker-compose.yml`

Как запустить всю систему вместе.

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .    # ← устанавливает проект как пакет

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**`-e .` (editable install)** — пакет установлен "на месте". В docker-compose монтируется `volumes: .:/app`, поэтому изменения кода сразу применяются без пересборки контейнера.

**`python:3.11-slim`** — минимальный Python-образ без лишних пакетов.

## docker-compose.yml — 7 сервисов

```
api ──────▶ chroma      (ChromaDB, порт 8000)
     ──────▶ qdrant      (Qdrant, порт 6333)
     ──────▶ ollama      (локальные LLM, порт 11434)
     ──────▶ jaeger      (трейсинг, UI: 16686, OTLP: 4317)

prometheus ──▶ api       (скрейп метрик /v1/metrics)
grafana ────▶ prometheus (дашборды)
```

### Все сервисы

| Сервис | Образ | Порты | Роль |
|---|---|---|---|
| `api` | *(build: .)* | 8080 | FastAPI-приложение |
| `chroma` | `chromadb/chroma:latest` | 8000 | Векторная БД Chroma |
| `qdrant` | `qdrant/qdrant:latest` | 6333 | Векторная БД Qdrant |
| `ollama` | `ollama/ollama:latest` | 11434 | Локальные LLM |
| `jaeger` | `jaegertracing/all-in-one:latest` | 16686 (UI), 4317 (OTLP) | Трейсинг |
| `prometheus` | `prom/prometheus:latest` | 9090 | Сбор метрик |
| `grafana` | `grafana/grafana:latest` | 3000 | Дашборды |

### Переопределение URL при Docker

В `.env` используются `localhost`-адреса (для запуска без Docker). В Docker-сети сервисы видят друг друга по именам, поэтому docker-compose переопределяет:

```yaml
environment:
  - CHROMA_HOST=chroma          # не localhost, а имя сервиса
  - QDRANT_URL=http://qdrant:6333
  - OLLAMA_BASE_URL=http://ollama:11434/v1
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
```

### Volumes — персистентные данные

```yaml
volumes:
  chroma_data:    # данные Chroma переживают перезапуск
  qdrant_data:    # данные Qdrant
  ollama_data:    # скачанные модели Ollama
  grafana_data:   # конфиг и дашборды Grafana
```

### Hot-reload для разработки

```yaml
api:
  volumes:
    - .:/app           # монтируем весь проект в контейнер
  command: uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

`--reload` — uvicorn перезапускается при изменении Python-файлов. Не нужно пересобирать контейнер при каждом изменении кода.

## Как запустить

### Полный стек (все сервисы)

```bash
docker-compose up -d
```

### Только для разработки (без Docker)

```bash
# Chroma локально (без Docker)
CHROMA_PERSIST_DIR=./chroma_data

# Или Qdrant в Docker
docker run -p 6333:6333 qdrant/qdrant

# API локально
uvicorn api.main:app --reload --port 8080
```

### Минимальный стек (API + Chroma)

```bash
docker-compose up -d chroma api
```

## Адреса после запуска

| Сервис | URL | Описание |
|---|---|---|
| FastAPI | http://localhost:8080 | API |
| Swagger UI | http://localhost:8080/docs | Интерактивная документация |
| Prometheus | http://localhost:9090 | Метрики |
| Grafana | http://localhost:3000 | Дашборды |
| Jaeger UI | http://localhost:16686 | Трейсы |
| Chroma | http://localhost:8000 | ChromaDB API |
| Qdrant | http://localhost:6333 | Qdrant API |
| Ollama | http://localhost:11434 | Ollama API |

## pyproject.toml — зависимости

Проект установлен как пакет. Основные зависимости:

```toml
[project]
dependencies = [
    "fastapi", "uvicorn",         # API-сервер
    "pydantic-settings",          # конфигурация
    "openai", "anthropic",        # LLM-провайдеры
    "sentence-transformers",      # локальные embedding-модели
    "chromadb", "qdrant-client",  # векторные БД
    "langchain-text-splitters",   # разбивка текста
    "tiktoken",                   # токенизация
    "tenacity",                   # retry
    "slowapi",                    # rate limiting
    "opentelemetry-*",            # трейсинг
    "prometheus-client",          # метрики
    "python-jose",                # JWT
    "mcp",                        # MCP-сервер
]

[project.optional-dependencies]
eval = ["ragas", "datasets"]      # pip install -e ".[eval]" для оценки
```

Опциональная группа `eval` — RAGAS и HuggingFace datasets. Не нужна в production, только для оценки.
