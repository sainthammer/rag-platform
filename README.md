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
