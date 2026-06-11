# RAG Platform

Платформа для построения RAG (Retrieval-Augmented Generation) пайплайнов с поддержкой MCP, оценки качества и наблюдаемости.

---

## Содержание

- [Что было реализовано](#что-было-реализовано)
- [Архитектура](#архитектура)
- [Модули](#модули)
- [REST API](#rest-api)
- [MCP-сервер](#mcp-сервер)
- [Наблюдаемость](#наблюдаемость)
- [Как читать Grafana dashboard](#как-читать-grafana-dashboard)
- [Как интерпретировать RAGAS отчёт](#как-интерпретировать-ragas-отчёт)
- [Сравнение chunk_size: 256 vs 512](#сравнение-chunk_size-256-vs-512)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Запуск](#запуск)
- [Тесты](#тесты)
- [Стек](#стек)
- [Hybrid Vector Search (BM25 + Qdrant + RRF)](#hybrid-vector-search-bm25--qdrant--rrf)

---

## Что было реализовано

### Hybrid Vector Search — BM25 + Qdrant + RRF

| Что добавлено | Файл | Описание |
|---|---|---|
| `BM25SparseVectorizer` | `vector_store/bm25.py` | Разреженный BM25-векторизатор: fit по корпусу, transform → `SparseVector`, поддержка кириллицы |
| `QdrantVectorStore` | `vector_store/adapters.py` | Реализация `VectorDB` поверх Qdrant: dense + sparse именованные векторы, auto-fit BM25 |
| `HybridVectorStore` | `vector_store/adapters.py` | RRF-fusion dense + BM25 sparse через `hybrid_search()` |
| `OllamaEmbeddingService` | `embeddings/adapters.py` | Embedding через Ollama `/v1/embeddings` (`nomic-embed-text`, 768 dim) |
| `reindex.py` | `reindex.py` | CLI: очистить коллекцию и переиндексировать документы из директории с прогресс-баром |
| Интеграционные тесты | `tests/integration/test_chroma_qdrant.py` | 14 in-memory + 3 testcontainers-теста: BM25, Chroma, Qdrant, Hybrid |
| Демо-скрипт | `demo.py` | 6 шагов: BM25 → dense → sparse → hybrid → reindex → полный RAG с `llama3.2` |

### REST API (`api/`)

| Endpoint | Файл | Описание |
|---|---|---|
| `POST /v1/ingest` | `api/routers/ingest.py` | Приём документа (файл / URL / текст), нарезка на чанки, embedding, запись в VectorDB. Задача выполняется асинхронно через `asyncio.create_task`. |
| `GET /v1/ingest/{job_id}` | `api/routers/ingest.py` | Опрос статуса и числа проиндексированных чанков. |
| `POST /v1/search` | `api/routers/search.py` | Семантический поиск по коллекции: embed запроса → `db.search()` → фильтрация по `score_threshold`. |
| `POST /v1/ask` | `api/routers/ask.py` | Полный RAG-цикл. При `stream=false` — JSON с `answer`, `sources`, `confidence`. При `stream=true` — SSE-поток токенов (`text/event-stream`). |
| `POST /v1/eval/run` | `api/routers/eval.py` | Запуск RAGAS-оценки в фоне. Принимает `mode` (`mock`/`ollama`), `max_cases`, `output_path`. |
| `GET /v1/eval/run/{job_id}` | `api/routers/eval.py` | Статус оценки, путь к HTML-отчёту, средние RAGAS-метрики. |
| `GET /v1/metrics` | `api/routers/metrics.py` | Prometheus-метрики через `generate_latest()`. Реализован как обычный роутер, а не `mount`. |

Вспомогательные модули, добавленные вместе с API:

- **`api/jobs.py`** — универсальное in-memory хранилище фоновых задач с `asyncio.Lock`. Используется и ingest, и eval роутерами.
- **`api/schemas.py`** — Pydantic-схемы для всех новых endpoints: `IngestJobResponse`, `SearchRequest`, `SearchResultItem`, `AskRequest`, `AskResponse`, `EvalRunRequest` и др.
- **`api/deps.py`** — добавлена зависимость `get_embed_service(request)`, читающая `app.state.embed_service`.
- **`chunking/__init__.py`** — модуль нарезки текста через `RecursiveCharacterTextSplitter`.

### MCP-сервер (`mcp/rag_server.py`)

| Что добавлено | Описание |
|---|---|
| Инструмент `ingest_document` | Принимает текст, режет на чанки, embedds, сохраняет в VectorDB. |
| Инструмент `ask` | Подключён к `RAGPipeline.ask()`, возвращает ответ с источниками и `confidence`. |
| `--transport http` | Новый режим запуска: `StreamableHTTPSessionManager` (stateless) + Starlette + Uvicorn. Endpoint: `/mcp`. |

### E2E-тест (`tests/e2e/test_api_flow.py`)

Тест поднимает полное FastAPI-приложение (без внешних сервисов) и проверяет:

1. Загрузка документа через `/v1/ingest` → фоновая задача завершается, `chunks_indexed > 0`.
2. Поиск по `/v1/search` возвращает результаты из только что проиндексированного текста.
3. `/v1/ask` возвращает непустой список `sources` (retrieval сработал корректно).
4. `/v1/ask` с `stream=true` возвращает `text/event-stream`.
5. `/v1/eval/run` возвращает `job_id` со статусом `pending`.

---

## Архитектура

```
rag-platform/
├── api/                    # FastAPI-приложение
│   ├── routers/            # Роутеры по домену (ingest, search, ask, eval, …)
│   ├── middleware/auth.py  # X-API-Key + JWT Bearer аутентификация
│   ├── deps.py             # FastAPI-зависимости (get_pipeline, get_embed_service, …)
│   ├── jobs.py             # In-memory хранилище фоновых задач
│   ├── schemas.py          # Pydantic-схемы запросов и ответов
│   └── app.py              # Точка сборки: lifespan, роутеры, OTel
│
├── chunking/               # Нарезка текста на чанки (RecursiveCharacterTextSplitter)
├── embeddings/             # EmbeddingService: OpenAI, SentenceTransformers, кэш, fake
├── vector_store/           # VectorDB ABC + ChromaDB / Qdrant адаптеры
├── retrieval/              # RAGPipeline: embed → search → generate
├── llm/                    # LLMProvider: OpenAI, Anthropic, Ollama + промпты + бюджет токенов
├── mcp/                    # MCP-сервер (stdio + Streamable HTTP)
├── evaluation/             # RAGAS-оценка: тест-кейсы, runner, HTML-отчёт
├── observability/          # OpenTelemetry трейсинг + Prometheus метрики
├── config.py               # Pydantic Settings + фабрики компонентов
└── tests/
    ├── unit/               # Тесты без внешних сервисов
    ├── integration/        # Тесты с реальным ChromaDB in-memory
    └── e2e/                # E2E-тест полного цикла ingest → search → ask
```

Поток данных при запросе `/v1/ask`:

```
POST /v1/ask
     │
     ├─▶ require_auth (X-API-Key / JWT)
     ├─▶ get_pipeline (app.state.pipeline)
     │
     ├─▶ RAGPipeline.ask(question, collection)
     │       ├─▶ embed_fn(question)          → вектор запроса
     │       ├─▶ vector_db_factory(collection)
     │       ├─▶ db.search(embedding, n)     → чанки + дистанции
     │       ├─▶ TokenBudgetManager          → усечение контекста
     │       ├─▶ PromptTemplate.format_*()   → системный / пользовательский промпт
     │       └─▶ LLMProvider.complete()      → ответ
     │
     └─▶ AskResponse { answer, sources, confidence, latency_ms }
         или StreamingResponse (SSE, text/event-stream)
```

---

## Модули

### `config.py`

Единая точка конфигурации через `pydantic-settings`. Читает переменные окружения и `.env`. Содержит три фабрики:

- `build_llm_provider(settings)` — создаёт `OpenAIProvider`, `AnthropicProvider` или `OllamaProvider`.
- `build_vector_db(settings, collection)` — создаёт `ChromaDB` (локальный или HTTP) или `QdrantDB`.
- `build_embedding_service(settings)` — создаёт `OpenAIEmbeddingService` или `SentenceTransformersService`, опционально оборачивает в `CachedEmbeddingService`.

### `embeddings/`

| Файл | Содержимое |
|---|---|
| `ports.py` | ABC `EmbeddingService` с методами `embed(text)` и `embed_batch(texts)` |
| `adapters.py` | `OpenAIEmbeddingService`, `SentenceTransformersService`, `FakeEmbeddingService` |
| `cache.py` | `CachedEmbeddingService` — SQLite-кэш поверх любого провайдера |
| `service.py` | `build_embedding_service()` — фабрика, читает `Settings` |
| `utils.py` | L2-нормализация векторов |

`FakeEmbeddingService` производит детерминированные векторы без ML-модели — используется в тестах.

### `vector_store/`

| Файл | Содержимое |
|---|---|
| `ports.py` | ABC `VectorDB`: `add`, `search`, `delete`, `count` |
| `adapters.py` | `ChromaDB` (PersistentClient / HttpClient), `QdrantDB` |
| `store_dataclasses.py` | `SearchResult(ids, documents, distances, metadatas)` |

### `llm/`

| Файл | Содержимое |
|---|---|
| `ports.py` | ABC `LLMProvider` с методом `complete(messages, stream)` |
| `adapters.py` | `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider` |
| `prompt_templates.py` | Шаблоны `BASE`, `STRICT`, `CITATION`, `MULTILINGUAL` |
| `token_budget.py` | `TokenBudgetManager` — усечение чанков под лимит токенов |
| `pipeline.py` | `RAGPipeline` с OTel-трейсингом и Prometheus метриками |

### `retrieval/`

`RAGPipeline` — основной класс пайплайна:

- `ask(question, collection)` → `RAGResponse` с полями `answer`, `sources`, `confidence`, `latency_ms`.
- `ask_stream(question, collection)` → `AsyncGenerator[str]` — потоковая генерация токен за токеном.
- `ask_multi(question, collections)` — параллельный поиск по нескольким коллекциям.
- `_retrieve(question, collection)` → список `SourceChunk` с полями `text`, `score`, `doc_id`, `metadata`.

### `chunking/`

```python
from chunking import chunk_text

chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
```

Использует `RecursiveCharacterTextSplitter` из `langchain-text-splitters`. Пустые чанки автоматически отфильтровываются.

### `api/`

#### Аутентификация (`middleware/auth.py`)

Зависимость `require_auth` поддерживает два метода:
- `X-API-Key: <key>` — прямое сравнение с `settings.api_secret_key`.
- `Authorization: Bearer <token>` — верификация JWT (алгоритм из `JWT_ALGORITHM`).

#### Lifespan (`app.py`)

При старте приложения в `app.state` кладутся:
- `embed_service` — `EmbeddingService`
- `pipeline` — `RAGPipeline`
- `llm`, `vector_db`, `store_client`, `store_backend`

Это позволяет роутерам получать компоненты через `Depends(get_pipeline)` без пересоздания на каждый запрос.

#### Фоновые задачи (`jobs.py`)

In-memory хранилище задач (`asyncio.Lock` для потокобезопасности):

```python
job = new_job()                          # создать, status="pending"
asyncio.create_task(_worker(job.job_id)) # запустить в фоне
job = get_job(job_id)                    # опросить статус
await update_job(job_id, "done", {...})  # обновить из воркера
```

Статусы: `pending → running → done | error`.

---

## REST API

Все endpoints требуют аутентификации, кроме `/v1/health` и `/v1/metrics`.

### `POST /v1/ingest` — загрузка документа

Принимает ровно один источник (form-data):

| Поле | Тип | Описание |
|---|---|---|
| `file` | UploadFile | Текстовый файл (UTF-8) |
| `url` | str | URL для загрузки через HTTP |
| `text` | str | Текст напрямую |
| `collection` | str | Целевая коллекция (default: `"default"`) |
| `doc_id` | str | Идентификатор документа |
| `chunk_size` | int | Размер чанка (50–4000, default: 500) |
| `chunk_overlap` | int | Перекрытие (0–500, default: 50) |

Возвращает `202 Accepted` с `job_id`. Индексация выполняется асинхронно:

```json
{ "job_id": "abc123", "status": "pending" }
```

### `GET /v1/ingest/{job_id}` — статус индексации

```json
{
  "job_id": "abc123",
  "status": "done",
  "chunks_indexed": 6,
  "collection": "default",
  "error": null
}
```

### `POST /v1/search` — семантический поиск

```json
{
  "query": "Python язык программирования",
  "collection": "default",
  "n_results": 5,
  "score_threshold": 0.0
}
```

Ответ:

```json
{
  "results": [
    { "id": "doc_chunk0", "text": "...", "score": 0.82, "metadata": {} }
  ]
}
```

### `POST /v1/ask` — полный RAG-цикл

```json
{
  "question": "Что такое Python?",
  "collection": "default",
  "stream": false,
  "score_threshold": 0.0,
  "n_results": 5
}
```

**stream=false** → JSON:

```json
{
  "answer": "Python — высокоуровневый язык...",
  "sources": [
    { "text": "...", "score": 0.87, "doc_id": "doc_chunk0", "metadata": {} }
  ],
  "confidence": 0.87,
  "latency_ms": 312.4
}
```

**stream=true** → `text/event-stream` (SSE):

```
data: {"token": "Python"}

data: {"token": " —"}

data: [DONE]
```

### `POST /v1/eval/run` — запуск RAGAS-оценки

```json
{ "mode": "mock", "max_cases": 10, "output_path": "eval_report.html" }
```

Возвращает `202` с `job_id`. Результат (HTML-отчёт, метрики RAGAS) доступен через:

```
GET /v1/eval/run/{job_id}
```

### `GET /v1/health` — состояние системы

```json
{
  "status": "ok",
  "vector_store": "chroma",
  "llm_provider": "ollama"
}
```

### `GET /v1/metrics` — Prometheus метрики

Возвращает метрики в формате `text/plain` для скрапинга Prometheus.

---

## MCP-сервер

`mcp/rag_server.py` реализует сервер по протоколу [Model Context Protocol](https://modelcontextprotocol.io).

### Инструменты

| Инструмент | Аргументы | Описание |
|---|---|---|
| `list_collections` | — | Список коллекций с числом векторов |
| `search` | `query`, `collection`, `n_results` | Семантический поиск |
| `ingest_document` | `text`, `collection`, `doc_id`, `chunk_size` | Индексировать документ |
| `ask` | `question`, `collection` | Полный RAG-цикл, возвращает ответ с источниками |

### Транспорты

**stdio** (для Claude Desktop):

```bash
PYTHONPATH=. .venv/bin/python mcp/rag_server.py
```

Конфиг `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-platform": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/rag-platform/mcp/rag_server.py"]
    }
  }
}
```

**HTTP** (Streamable HTTP, stateless):

```bash
PYTHONPATH=. .venv/bin/python mcp/rag_server.py --transport http --port 8001
```

Endpoint: `http://localhost:8001/mcp`

Реализовано через `StreamableHTTPSessionManager` из `mcp.server.streamable_http_manager`, обёрнутый в Starlette-приложение с lifespan.

---

## Наблюдаемость

### OpenTelemetry (`observability/tracing.py`)

```python
from observability.tracing import setup_tracing, instrument_fastapi

setup_tracing("rag-platform")   # настройка TracerProvider + OTLP gRPC экспортёр
instrument_fastapi(app)         # авто-инструментация FastAPI
```

Трейсы экспортируются на `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://localhost:4317`).

### Prometheus (`observability/metrics.py`)

Доступные метрики:

| Метрика | Тип | Описание |
|---|---|---|
| `rag_requests_total` | Counter | Число RAG-запросов |
| `rag_latency_seconds` | Histogram | Латентность полного цикла |
| `embedding_requests_total` | Counter | Число embedding-запросов |
| `embedding_latency_seconds` | Histogram | Латентность embedding |
| `llm_requests_total` | Counter | Число LLM-запросов |
| `llm_latency_seconds` | Histogram | Латентность LLM |
| `retrieval_chunk_count` | Histogram | Число чанков, найденных при retrieval |

Метрики доступны на `GET /v1/metrics`.

### Docker Compose (`docker-compose.yml`)

Включает: Jaeger (трейсинг), Prometheus (метрики), Grafana (дашборды), Ollama (LLM), ChromaDB.

```bash
docker-compose up --build
docker compose exec ollama ollama pull llama3.2
```

---

## Как читать Grafana dashboard

Дашборд `RAG Platform` (http://localhost:3000, `admin/admin`) загружается автоматически через provisioning. Он разбит на **4 секции**:

| Секция | Что показывает |
|---|---|
| **Overview** | Ключевые KPI: RPS, процент ошибок, latency P50/P95, общее число запросов + временны́е ряды |
| **RAG Quality** | Качество retrieval: среднее число чанков на запрос, распределение по перцентилям |
| **Cost Tracker** | Нагрузка на LLM и Embedding: кол-во запросов по провайдерам и их latency |
| **Infrastructure** | Детальный мониторинг компонентов по провайдеру/модели/статусу |

### Ключевые пороги и сигналы

**Overview**

- `RAG — процент ошибок` — зелёный < 5 %, жёлтый 5–20 %, красный > 20 %.
- `RAG — latency P95` — типичные значения: Ollama llama3.2 ≈ 1–5 сек, OpenAI gpt-4o-mini < 1 сек.

**RAG Quality**

- `Чанков на запрос (среднее)` — норма 3–5. Значение < 2 сигнализирует о разреженной базе; > 8 — контекст будет обрезан `TokenBudgetManager`.
- `Распределение числа чанков (P99)` — «худший случай» retrieval; используйте для выбора `n_results`.

**Cost Tracker**

- `LLM — ошибки по провайдеру` — цветовая индикация: зелёный 0, жёлтый ≥ 1, красный ≥ 10.
- `Embedding latency P95` — SentenceTransformers ≈ 50–200 мс; OpenAI < 200 мс.

### Фильтры времени

Диапазон выбирается в правом верхнем углу:
- `Last 5 minutes` — при отладке в реальном времени.
- `Last 30 minutes` — стандартный мониторинг (default).
- `Last 6 hours` — анализ тренда нагрузки.

---

## Как интерпретировать RAGAS отчёт

Отчёт создаётся командой `python evaluation/eval_runner.py` или через `POST /v1/eval/run`.  
Выходной файл: `eval_report.html` (открывается в браузере).

### Метрики RAGAS

| Метрика | Что измеряет | Хорошо | Плохо |
|---|---|---|---|
| **Faithfulness** | Ответ не придумывает факты, которых нет в retrieved-чанках | ≥ 0.8 | < 0.5 → модель галлюцинирует |
| **Answer Relevancy** | Ответ отвечает именно на поставленный вопрос | ≥ 0.7 | < 0.4 → ответ уходит в сторону |
| **Context Precision** | Доля retrieved чанков, реально нужных для ответа | ≥ 0.6 | < 0.4 → retrieval «шумит» |
| **Context Recall** | Доля нужных фактов, найденных в retrieved чанках | ≥ 0.6 | < 0.4 → retrieval не находит документы |

### Цветовые коды

| Цвет полосы | Значение | Диапазон |
|---|---|---|
| Зелёный | Хорошо | score ≥ 0.7 |
| Жёлтый | Допустимо | 0.4 ≤ score < 0.7 |
| Красный | Требует внимания | score < 0.4 |

### Секция «Тест на галлюцинации»

Проверяет **negative-кейсы** — вопросы, ответов на которые нет в базе знаний (например: «Какой встроенный модуль Python даёт доступ к GPU через CUDA?»).

- **PASS** — модель ответила «не нашёл информации» / «недостаточно данных». Правильное поведение.
- **FAIL** — модель придумала ответ. В mock-режиме FAIL ожидаем; в Ollama-режиме с шаблоном `STRICT` должен быть PASS.

### Диагностика по результатам

| Наблюдение | Возможная причина | Что проверить |
|---|---|---|
| Faithfulness < 0.5 | Модель игнорирует контекст | Шаблон промпта (`STRICT`), температура LLM |
| Context Recall < 0.4 | Retrieval не находит нужные чанки | `chunk_size`, модель embeddings, `n_results` |
| Context Precision < 0.4 | Retrieval тянет нерелевантные чанки | `score_threshold`, уменьшить `n_results` |
| Все метрики N/A | Запуск в mock-режиме | Используйте `--mode ollama` для реальных баллов |

### Запуск оценки

```bash
# Offline, без внешних сервисов (mock LLM — RAGAS не считается)
python evaluation/eval_runner.py

# Ollama + RAGAS (требует ollama serve)
python evaluation/eval_runner.py --mode ollama

# Больше кейсов для RAGAS (точнее, но дольше)
python evaluation/eval_runner.py --mode ollama --max-cases 20

# Через REST API
curl -X POST http://localhost:8080/v1/eval/run \
     -H "X-API-Key: $API_SECRET_KEY" \
     -H "Content-Type: application/json" \
     -d '{"mode": "mock", "max_cases": 5, "output_path": "eval_report.html"}'
```

---

## Сравнение chunk_size: 256 vs 512

Скрипт `compare_chunks.py` индексирует выборку документов в две временные
коллекции (`bench_256` / `bench_512`) и прогоняет тестовые вопросы через
retrieval, измеряя качество и скорость.

### Запуск

```bash
# Базовый запуск (100 файлов, 15 вопросов)
docker compose exec api python compare_chunks.py

# Больше файлов для точности (займёт дольше)
docker compose exec api python compare_chunks.py --max-files 300 --max-questions 15

# Сохранить результаты в кастомный путь
docker compose exec api python compare_chunks.py --output results/my_report.json
```

Параметры:

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--max-files` | 100 | Кол-во `.md` файлов для индексации |
| `--max-questions` | 15 | Кол-во тестовых вопросов |
| `--docs-dir` | `docs` | Путь к директории с документами |
| `--output` | `results/chunk_comparison.json` | Путь для сохранения JSON-результатов |

### Что измеряется

| Метрика | Что показывает |
|---|---|
| **Чанков проиндексировано** | Меньший chunk_size → больше чанков (выше покрытие, больше памяти) |
| **Средняя длина чанка** | Прямое следствие chunk_size |
| **Средний confidence** | Max retrieval score по вопросу: выше = релевантнее совпадение |
| **Среднее кол-во источников** | Сколько чанков с ненулевым score возвращается |
| **Средняя / P95 latency** | Время embed + search (без LLM) |

### Как интерпретировать

- **chunk_size=256**: больше чанков, более точные совпадения по конкретным фактам.
  Хорошо для точечных вопросов («что такое X?»).
- **chunk_size=512**: меньше чанков, каждый чанк содержит больше контекста.
  Хорошо для вопросов, требующих развёрнутого ответа.
- Если `avg_confidence` заметно выше у одного из вариантов — его стоит предпочесть.
- Если разница < 0.01 — размер чанка не критичен для этой коллекции.

После запуска скрипт выводит таблицу в консоль и сохраняет JSON с полными результатами.
Временные коллекции (`bench_256`, `bench_512`) остаются в ChromaDB и не влияют на `default`.

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd rag-platform

# 2. Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS

# 3. Установить зависимости
pip install -e ".[dev]"

# Опционально: зависимости для RAGAS-оценки
pip install -e ".[dev,eval]"

# 4. Настроить конфиг
cp .env.example .env
# Отредактировать .env
```

---

## Конфигурация

Все параметры задаются через `.env` или переменные окружения.

### LLM-провайдер

```bash
# Ollama (локально, ключ не нужен)
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

### Embeddings

```bash
EMBEDDING_PROVIDER=sentence-transformers   # или openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_NORMALIZE=true
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_PATH=.cache/embeddings.sqlite3
```

### Vector Store

```bash
VECTOR_STORE_BACKEND=chroma   # или qdrant

# ChromaDB — локальный файловый режим (без сервера)
CHROMA_PERSIST_DIR=./chroma_data

# ChromaDB — HTTP-режим
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Qdrant
QDRANT_URL=http://localhost:6333
```

### API

```bash
API_HOST=0.0.0.0
API_PORT=8080
API_SECRET_KEY=change-me-in-production   # ОБЯЗАТЕЛЬНО сменить
JWT_ALGORITHM=HS256
```

---

## Запуск

### Локально

```bash
# API
PYTHONPATH=. uvicorn api.main:app --reload --port 8080

# MCP stdio
PYTHONPATH=. .venv/bin/python mcp/rag_server.py

# MCP HTTP
PYTHONPATH=. .venv/bin/python mcp/rag_server.py --transport http --port 8001
```

### Docker

```bash
docker-compose up --build
# Swagger UI: http://localhost:8080/docs
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000
# Jaeger:      http://localhost:16686
```

### Примеры модулей

```bash
# Embeddings
PYTHONPATH=. .venv/bin/python -m embeddings.example fake

# Vector Store
PYTHONPATH=. .venv/bin/python vector_store/example.py

# LLM + RAGPipeline
PYTHONPATH=. .venv/bin/python llm/example.py ollama

# Observability
PYTHONPATH=. .venv/bin/python observability/example.py
```

---

## Тесты

```bash
# Все тесты
PYTHONPATH=. .venv/bin/python -m pytest

# Только unit (без внешних сервисов, быстро)
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/ -v

# E2E-тест: ingest → search → ask → проверка sources
PYTHONPATH=. .venv/bin/python -m pytest tests/e2e/ -v

# С покрытием
PYTHONPATH=. .venv/bin/python -m pytest --cov=. --cov-report=html
```

### Что тестирует E2E-тест (`tests/e2e/test_api_flow.py`)

1. `test_ingest_search_ask_flow` — полный цикл:
   - POST `/v1/ingest` (текст) → `202 + job_id`
   - GET `/v1/ingest/{job_id}` — ждёт `status=done`, проверяет `chunks_indexed > 0`
   - POST `/v1/search` — проверяет, что результаты есть
   - POST `/v1/ask` — проверяет, что `sources` не пусты и `confidence > 0`

2. `test_ask_stream_returns_sse` — `/ask` с `stream=true` возвращает `text/event-stream`.

3. `test_ingest_job_not_found` — GET `/v1/ingest/nonexistent` → `404`.

4. `test_ingest_no_source_returns_422` — POST без источника → `422`.

5. `test_eval_run_returns_job_id` — POST `/v1/eval/run` → `202 + job_id`.

Тест использует `FakeEmbeddingService` (детерминированные векторы), `_MockLLM` и ChromaDB в `tmp_path`. Lifespan не запускается через `ASGITransport` — `app.state` заполняется вручную.

---

## Стек

| Компонент | Технология |
|---|---|
| API | FastAPI + Uvicorn |
| Аутентификация | python-jose (X-API-Key + JWT) |
| Vector Store | ChromaDB / Qdrant (dense + sparse + hybrid RRF) |
| Embeddings | OpenAI / SentenceTransformers / Ollama / кэш SQLite |
| Chunking | langchain-text-splitters |
| LLM | OpenAI / Anthropic / Ollama |
| MCP | mcp SDK (stdio + Streamable HTTP) |
| Evaluation | RAGAS + langchain-ollama |
| Трейсинг | OpenTelemetry + Jaeger |
| Метрики | Prometheus + Grafana |
| Конфигурация | pydantic-settings |
| Тесты | pytest + pytest-asyncio + httpx |

---

## Hybrid Vector Search (BM25 + Qdrant + RRF)

### Что добавлено

#### `vector_store/bm25.py` (новый)

BM25-векторизатор для разреженного индекса Qdrant.

| Класс | Описание |
|---|---|
| `SparseVector` | Разреженный вектор: параллельные массивы `indices` + `values` |
| `BM25SparseVectorizer` | Fit по корпусу → `transform(text)` → `SparseVector` с BM25-весами |

Формула IDF: `log((N − df + 0.5) / (df + 0.5) + 1)`. Параметры: `k1=1.5`, `b=0.75`.
Токенизатор поддерживает латиницу, кириллицу и цифры.

```python
from vector_store.bm25 import BM25SparseVectorizer

v = BM25SparseVectorizer()
v.fit(corpus)
sv = v.transform("BM25 ранжирование термин")
# sv.indices, sv.values — готово для Qdrant sparse vector
```

#### `vector_store/adapters.py` — новые классы

**`QdrantVectorStore(VectorDB)`** — реализует тот же ABC, что и `ChromaDB`:

- Именованные вектора Qdrant: `dense` (cosine) + `sparse` (BM25).
- `add()` — авто-fit BM25 при первом вызове, upsert с обоими векторами.
- `search()` — dense cosine-поиск (совместим с `VectorDB`).
- `sparse_search(query_text)` — BM25 keyword-поиск.
- `in_memory=True` — режим без внешнего сервиса (для тестов).

**`HybridVectorStore(QdrantVectorStore)`**:

- `hybrid_search(query_text, query_embedding, n_results, rrf_k=60)` — RRF-fusion dense + sparse.
- RRF: `score(d) = Σ 1 / (k + rank)`, 1-based ranks, `k=60`.

```python
from vector_store import HybridVectorStore, BM25SparseVectorizer

vectorizer = BM25SparseVectorizer().fit(corpus)
store = HybridVectorStore("my_col", vector_size=768, vectorizer=vectorizer,
                          host="localhost", port=6333)
store.add(ids, embeddings, documents, metadatas)

res = store.hybrid_search("ранжирование термин", query_embedding, n_results=5)
```

#### `embeddings/adapters.py` — `OllamaEmbeddingService` (новый)

Embedding-сервис через Ollama OpenAI-совместимый API (`/v1/embeddings`).
Рекомендуемая модель: `nomic-embed-text` (768 dim, поддерживает русский язык).

```python
from embeddings.adapters import OllamaEmbeddingService

svc = OllamaEmbeddingService(model="nomic-embed-text",
                             base_url="http://localhost:11434/v1")
vec = svc.embed("векторная база данных")  # list[float], dim=768
```

#### `reindex.py` (новый) — CLI переиндексации

Очищает коллекцию и переиндексирует документы из директории с прогресс-баром.

```bash
# Qdrant + Ollama embeddings
python reindex.py ./docs \
    --backend qdrant \
    --collection my_docs \
    --embedding-provider ollama \
    --ollama-model nomic-embed-text \
    --vector-size 768

# ChromaDB + fake embeddings (dev/test)
python reindex.py ./docs --backend chroma --collection my_docs
```

Поддерживаемые форматы: `.txt`, `.md`, `.pdf`, `.docx`, `.html`.
Параметры `--chunk-size`, `--chunk-overlap`, `--batch-size`.

#### `tests/integration/test_chroma_qdrant.py` (новый)

14 in-memory тестов (без Docker) + 3 testcontainers-теста (требуют Docker).

| Группа | Что тестирует |
|---|---|
| BM25 | fit/transform, OOV, не обученный векторизатор |
| ChromaDB | exact-match (L2-distance < 0.1), metadata filter |
| Qdrant in-memory | exact-match (cosine > 0.9), sparse_search, delete/count, upsert idempotency |
| Hybrid | RRF scores descending, VectorDB совместимость |
| testcontainers | exact-match, hybrid_search, Chroma vs Qdrant overlap через реальный Docker |

```bash
# in-memory (быстро, без Docker)
PYTHONPATH=. pytest tests/integration/test_chroma_qdrant.py -v

# с Docker (testcontainers)
PYTHONPATH=. pytest tests/integration/test_chroma_qdrant.py -v -m integration
```

### Настройка Qdrant + Ollama

```bash
# 1. Запустить Qdrant
docker compose up -d qdrant

# 2. Установить Ollama и скачать модели
ollama pull nomic-embed-text   # 274 MB — embedding
ollama pull llama3.2           # LLM для генерации ответов

# 3. Конфиг .env
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

### Демо-скрипт (`demo.py`)

Интерактивное демо всего функционала с реальными сервисами (Ollama + Qdrant).

```bash
.venv/bin/python demo.py            # все 6 шагов
.venv/bin/python demo.py --step 2   # конкретный шаг
```

| Шаг | Что демонстрирует |
|---|---|
| 1 | BM25SparseVectorizer: токенизация, IDF-веса, sparse-вектора |
| 2 | Dense-поиск ChromaDB vs Qdrant с `nomic-embed-text` — правильная семантика русских запросов |
| 3 | Sparse BM25 vs Dense: keyword-запросы выигрывают у sparse |
| 4 | HybridVectorStore RRF: dense + BM25 через Qdrant Docker |
| 5 | `reindex.py`: очистка + переиндексация с Ollama embeddings |
| 6 | Полный RAG-пайплайн: ingest → hybrid search → `llama3.2` генерирует ответ |
