# RAG Platform

Платформа для построения RAG (Retrieval-Augmented Generation) пайплайнов с поддержкой MCP, оценки качества и наблюдаемости.

## Архитектура

```
rag_platform/
├── api/            # FastAPI routers, middleware, auth, rate limiting
├── chunking/       # Document loaders, chunking strategies
├── embeddings/     # EmbeddingService, cache, normalization
├── vector_store/   # VectorStore ABC + ChromaDB / Qdrant impls
├── retrieval/      # Retriever, re-ranker, MMR, HyDE
├── llm/            # RAGPipeline, prompt templates, token budget
├── mcp/            # MCP server (stdio + HTTP)
├── evaluation/     # RAGAS suite, test dataset, reports
├── observability/  # OTel, Prometheus, Grafana dashboards
└── config.py       # Pydantic Settings
tests/
├── unit/
├── integration/
└── e2e/
```

Подробная документация по модулю LLM: [`llm/README.md`](llm/README.md)

Документация по модулю Embeddings: [`embeddings/README.md`](embeddings/README.md)

Документация по оценке (RAGAS): [`evaluation/README.md`](evaluation/README.md)

Документация по трейсингу (OpenTelemetry): [`observability/README.md`](observability/README.md)

Подробная документация по модулю VECTOR STORE: [`vector_store/README.md`](vectore_store/README.md)

---

## Установка

### Требования

- Python 3.11+
- Docker & Docker Compose (для запуска через контейнеры)

### Шаги

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd rag-platform

# 2. Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Установить зависимости (основные + dev-инструменты)
pip install -e ".[dev]"

# 4. Скопировать конфиг и выбрать провайдер
cp .env.example .env
```

---

## Конфигурация

Все настройки задаются через переменные окружения или файл `.env`.
Полный список параметров см. в `.env.example` и `config.py`.

### Embeddings

Модуль `embeddings/` отвечает за построение embedding-векторов, L2-нормализацию
и SQLite-кэш. Провайдер выбирается через `.env`:

```bash
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_NORMALIZE=true
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_PATH=.cache/embeddings.sqlite3
```

Доступные провайдеры:
- `openai` — OpenAI Embeddings API;
- `sentence-transformers` — локальная модель через `sentence-transformers`.

### Evaluation (RAGAS)

Модуль `evaluation/` содержит фиксированный набор тесткейсов (30 positive / 10 negative / 5 multi-hop)
и утилиты для оценки качества через RAGAS.

Путь к базе знаний для `source_doc` задаётся переменной окружения:

```bash
KNOWLEDGE_BASE_PATH=/path/to/knowledge-base
```

Зависимости для RAGAS находятся в optional группе `eval`:

```bash
pip install -e ".[eval]"
```

### Observability / Tracing (OpenTelemetry)

Для трейсинга используется `observability/tracing.py`.
Экспорт трейсинга выполняется через OTLP endpoint:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Этот endpoint должен указывать на Jaeger Collector / OpenTelemetry Collector.

### Выбор LLM-провайдера

Отредактируйте `.env` и установите нужный провайдер:

```bash
# Ollama — локальная модель, ключ не нужен
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...
```

### Настройка Ollama (локальные модели)

Ollama позволяет запускать LLM локально без API-ключей.

```bash
# Установить Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Скачать модель (~2 GB)
ollama pull llama3.2

# Сервер запускается автоматически при установке.
# Если не запущен — стартовать вручную:
ollama serve
```

**Системные требования:**

| Модель | RAM | GPU |
|---|---|---|
| llama3.2 (3B) | 4 GB | не обязателен |
| llama3.2 (7B) | 8 GB | желателен |

---

## Запуск

### Smoke-тест LLM-модуля (без внешних сервисов)

Запускает юнит-тесты с mock-провайдером — API-ключи не нужны:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/ -v
```

### Интерактивный пример провайдеров и RAGPipeline

```bash
# Ollama (нужен запущенный сервер и скачанная модель)
PYTHONPATH=. .venv/bin/python llm/example.py ollama

# OpenAI (нужен ключ в .env)
PYTHONPATH=. .venv/bin/python llm/example.py openai

# Anthropic (нужен ключ в .env)
PYTHONPATH=. .venv/bin/python llm/example.py anthropic
```

Пример демонстрирует:
- прямой вызов провайдера (обычный и стриминг),
- полный RAG-цикл со всеми шаблонами промптов (`base`, `strict`, `citation`, `multilingual`),
- стриминг ответа через `RAGPipeline`.

### Пример vector_store

```bash
PYTHONPATH=. .venv/bin/python vector_store/example.py
```

Демонстрирует `add`, `search`, `delete`, `count` на ChromaDB и Qdrant in-memory.

### Smoke-тест embeddings

Проверка `EmbeddingService`, нормализации, batch-вызова и SQLite-кэша.
По умолчанию используется `fake`-провайдер без внешних сервисов:

```bash
PYTHONPATH=. .venv/bin/python -m embeddings.example
```

Доступные режимы:

```bash
# Детерминированный fake-провайдер, сеть и API-ключи не нужны
PYTHONPATH=. .venv/bin/python -m embeddings.example fake

# OpenAI Embeddings API, нужен OPENAI_API_KEY
PYTHONPATH=. .venv/bin/python -m embeddings.example openai

# Локальная/скачиваемая модель SentenceTransformers, например BAAI/bge-m3
PYTHONPATH=. .venv/bin/python -m embeddings.example sentence-transformers
```

Если SentenceTransformers-модель уже скачана, но HuggingFace недоступен:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m embeddings.example sentence-transformers
```

Пример выводит диагностические значения embedding-модуля:
- размерность вектора;
- L2-норму;
- первые 8 координат первого вектора;
- cosine similarity между текстами.

Unit-тесты модуля embeddings:

```bash
# Все unit-тесты embeddings
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_embeddings.py tests/unit/test_embedding_cache.py -q

# Только базовый контракт EmbeddingService: dimension, normalize, batch
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_embeddings.py -q

# Только SQLite-кэш: key, roundtrip, cache hit, missing-only batch
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_embedding_cache.py -q
```

Проверка стиля и статических правил для embeddings:

```bash
PYTHONPATH=. .venv/bin/python -m ruff check embeddings tests/unit/test_embeddings.py tests/unit/test_embedding_cache.py
```

### Запуск через Docker

```bash
docker-compose up --build
```

### Запуск API локально

```bash
uvicorn api.main:app --reload
```

---

## Тесты

```bash
# Все тесты
PYTHONPATH=. .venv/bin/python -m pytest

# Только unit (без внешних сервисов)
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/

# С отчётом о покрытии
PYTHONPATH=. .venv/bin/python -m pytest --cov=. --cov-report=html
```

---

## Стек

| Компонент     | Технология                              |
|---------------|-----------------------------------------|
| API           | FastAPI                                 |
| Vector Store  | ChromaDB / Qdrant                       |
| Embeddings    | OpenAI / HuggingFace                    |
| LLM           | OpenAI / Anthropic / Ollama             |
| Evaluation    | RAGAS                                   |
| Observability | OpenTelemetry, Prometheus, Grafana      |
| MCP           | Model Context Protocol                  |
