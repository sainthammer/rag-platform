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

## Быстрый старт

### Требования

- Python 3.11+
- Docker & Docker Compose

### Установка

```bash
# Клонировать репозиторий
git clone <repo-url>
cd rag_platform

# Установить зависимости
pip install -e ".[dev]"

# Скопировать конфиг
cp .env.example .env
# Заполнить .env своими значениями
```

### Запуск через Docker

```bash
docker-compose up --build
```

### Запуск локально

```bash
uvicorn api.main:app --reload
```

## Тесты

```bash
# Все тесты
pytest

# Только unit
pytest tests/unit

# С покрытием
pytest --cov=rag_platform --cov-report=html
```

## Конфигурация

Все настройки через переменные окружения или `.env` файл. См. `config.py` и `.env.example`.

## Стек

| Компонент      | Технология              |
|----------------|-------------------------|
| API            | FastAPI                 |
| Vector Store   | ChromaDB / Qdrant       |
| Embeddings     | OpenAI / HuggingFace    |
| LLM            | OpenAI / Anthropic      |
| Evaluation     | RAGAS                   |
| Observability  | OpenTelemetry, Prometheus, Grafana |
| MCP            | Model Context Protocol  |
