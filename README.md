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
- [Быстрый старт (5 шагов)](#быстрый-старт-5-шагов)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Запуск](#запуск)
- [curl-примеры](#curl-примеры)
- [Тесты](#тесты)
- [Стек](#стек)
- [Hybrid Vector Search (BM25 + Qdrant + RRF)](#hybrid-vector-search-bm25--qdrant--rrf)
- [Как добавить новый LLM-провайдер](#как-добавить-новый-llm-провайдер)
- [Chunking: стратегии и загрузчики](#chunking-стратегии-и-загрузчики)
- [Retrieval: переранжирование и TTL-кэш](#retrieval-переранжирование)
- [Коллекции API](#коллекции-api)

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

### Логирование RAG-вызовов (`retrieval/pipeline.py`)

Каждый вызов `RAGPipeline.ask()` пишет структурированную запись в логгер `rag.pipeline`:

| Поле | Что |
|---|---|
| `question` | первые 200 символов вопроса |
| `collection` | имя коллекции |
| `chunks_count` | число retrieved чанков |
| `prompt_tokens` | токены system + user сообщений |
| `completion_tokens` | токены в ответе LLM |
| `latency_ms` | полное время цикла |
| `cache_hit` | `true` / `false` |

При cache hit пишется отдельная запись `rag_cache_hit`. При fallback (нет контекста) — `rag_call` с нулевыми счётчиками токенов.

### TTL-кэш ответов (`retrieval/pipeline.py`)

`RAGPipeline` получил встроенный in-memory кэш с TTL 5 минут:
- Параметр `cache_ttl: float = 300.0` (0 — отключить)
- Ключ: `(question, collection)` — один вопрос в разных коллекциях хранится отдельно
- При cache hit LLM и embed не вызываются
- `ask_stream()` кэш не использует

### Rate Limiting (`api/limiter.py`)

Ограничение запросов через `slowapi` по IP клиента:

| Endpoint | Лимит |
|---|---|
| `POST /v1/ask` | 100 запросов / минуту |
| `POST /v1/search` | 500 запросов / минуту |

При превышении возвращается `HTTP 429 Too Many Requests`.

### Request ID Middleware (`api/middleware/request_id.py`)

Каждый запрос получает сквозной идентификатор `X-Request-ID`:
- Если клиент прислал `X-Request-ID` — используется он (сквозная трассировка)
- Иначе генерируется `uuid.uuid4()`
- Доступен в роутерах через `request.state.request_id`
- Прокидывается в OTel-спан как атрибут `http.request_id`
- Возвращается клиенту в заголовке ответа

### OpenAPI + Pydantic examples (`api/schemas.py`)

Все Pydantic-схемы дополнены:
- `description=` для каждого поля — видно в Swagger UI
- `examples=[]` — конкретные значения в документации
- `model_config` с `json_schema_extra` — полные примеры объектов
- Описания `429` в endpoints `/ask` и `/search`

### docker-compose.yml

| Улучшение | Детали |
|---|---|
| `restart: unless-stopped` | Автоперезапуск для всех сервисов |
| `healthcheck` | HTTP-проверки для api, chroma, qdrant |
| `depends_on` с `condition: service_healthy` | api стартует только после готовности chroma и qdrant |
| Grafana credentials | `GRAFANA_USER` / `GRAFANA_PASSWORD` из env |
| `:ro` монтирование | Конфиги prometheus и grafana монтируются read-only |

### Тесты (`tests/unit/`)

| Файл | Что тестирует |
|---|---|
| `test_llm_adapters.py` | OpenAI, Anthropic, Ollama провайдеры без реальных API (17 тестов) |
| `test_rag_cache_and_logging.py` | TTL-кэш (hit/miss/expiry/disabled) и логирование (13 тестов) |
| `test_pipeline.py` (дополнен) | `run_detailed()`, error propagation, `TokenBudgetManager.remaining()` |

Покрытие модуля `llm/` поднято с **55% → 95%**.

---

## Архитектура

```
rag-platform/
├── api/                         # FastAPI-приложение
│   ├── routers/
│   │   ├── ingest.py            # POST /v1/ingest, GET /v1/ingest/{job_id}
│   │   ├── search.py            # POST /v1/search
│   │   ├── ask.py               # POST /v1/ask (JSON + SSE streaming)
│   │   ├── eval.py              # POST /v1/eval/run, GET /v1/eval/run/{job_id}
│   │   ├── collections.py       # GET /v1/collections, DELETE /v1/collections/{name}
│   │   ├── health.py            # GET /v1/health
│   │   └── metrics.py           # GET /v1/metrics (Prometheus)
│   ├── middleware/
│   │   ├── auth.py              # X-API-Key + JWT Bearer аутентификация
│   │   └── request_id.py        # X-Request-ID: uuid4, OTel span attribute
│   ├── limiter.py               # slowapi Limiter (100/мин /ask, 500/мин /search)
│   ├── deps.py                  # FastAPI-зависимости (get_pipeline, get_embed_service, …)
│   ├── jobs.py                  # In-memory хранилище фоновых задач (asyncio.Lock)
│   ├── schemas.py               # Pydantic-схемы с description + examples для Swagger UI
│   ├── app.py                   # FastAPI-приложение: lifespan, роутеры, OTel, _seed_docs_task
│   └── main.py                  # Точка входа для uvicorn (re-export из app.py)
│
├── chunking/                    # Нарезка текста на чанки
│   ├── ports.py                 # ABC: Chunker, DocumentLoader, Chunk
│   ├── adapters.py              # FixedSizeChunker, ByHeaderChunker, SemanticChunker
│   ├── loaders.py               # TextLoader, MarkdownLoader, HTMLLoader, PDFLoader
│   ├── pipeline.py              # ingest(path) — маршрутизация по расширению файла
│   └── ingest.py                # CLI: chunk_text → JSON или ChromaDB
│
├── embeddings/                  # EmbeddingService
│   ├── ports.py                 # ABC EmbeddingService (embed, embed_batch)
│   ├── adapters.py              # OpenAI, SentenceTransformers, Ollama, Fake
│   ├── cache.py                 # CachedEmbeddingService (SQLite)
│   ├── service.py               # build_embedding_service() — фабрика
│   └── utils.py                 # L2-нормализация
│
├── vector_store/                # VectorDB
│   ├── ports.py                 # ABC VectorDB (add, search, delete, count)
│   ├── adapters.py              # ChromaDB, QdrantVectorStore, HybridVectorStore
│   ├── bm25.py                  # SparseVector, BM25SparseVectorizer
│   └── store_dataclasses.py     # SearchResult dataclass
│
├── retrieval/                   # RAG пайплайн
│   ├── ports.py                 # ABC Reranker (rerank)
│   ├── pipeline.py              # RAGPipeline, SourceChunk, RAGResponse
│   └── rerankers.py             # CrossEncoderReranker, MMRReranker
│
├── llm/                         # LLM провайдеры
│   ├── ports.py                 # ABC LLMProvider (complete)
│   ├── adapters.py              # OpenAIProvider, AnthropicProvider, OllamaProvider
│   ├── prompt_templates.py      # BASE, STRICT, CITATION, MULTILINGUAL шаблоны
│   ├── token_budget.py          # TokenBudgetManager
│   ├── pipeline.py              # RAGPipeline с OTel + Prometheus
│   └── llm_dataclasses.py       # Message(role, content)
│
├── mcp/rag_server.py            # MCP-сервер (stdio + Streamable HTTP)
│
├── evaluation/                  # RAGAS-оценка
│   ├── eval_runner.py           # run_evaluation(), generate_html_report()
│   ├── testcase.py              # TestCase dataclass
│   ├── testcases_dataset.py     # 45 тест-кейсов (positive/negative/multi_hop)
│   └── ragas_eval.py            # Интеграция с RAGAS API
│
├── observability/               # OpenTelemetry + Prometheus
│   ├── tracing.py               # setup_tracing(), instrument_fastapi()
│   └── metrics.py               # Prometheus counters, histograms
│
├── config.py                    # Pydantic Settings + три фабрики компонентов
├── reindex.py                   # CLI: очистить коллекцию и переиндексировать директорию
├── compare_chunks.py            # Бенчмарк chunk_size=256 vs chunk_size=512
├── demo.py                      # Интерактивное демо 6 шагов (BM25 → RAG)
├── .env.example                 # Шаблон переменных окружения
├── docker-compose.yml           # api, chroma, qdrant, ollama, jaeger, prometheus, grafana
└── tests/
    ├── unit/                    # Тесты без внешних сервисов (~1800 строк)
    ├── integration/             # ChromaDB in-memory + testcontainers-тесты
    └── e2e/                     # E2E-тест полного цикла ingest → search → ask
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
| `llm_dataclasses.py` | `Message(role, content)` dataclass |

#### Шаблоны промптов (`prompt_templates.py`)

`PromptTemplate` — frozen dataclass. Реестр через `get_template(name)`.

| Шаблон | Поведение | Когда использовать |
|---|---|---|
| `BASE` | Допускает факты из памяти модели; отвечает **только по-русски** | Общий вопрос-ответ |
| `STRICT` | Только по переданному контексту, без внешних знаний | Снижение галлюцинаций |
| `CITATION` | Ответ с цитатами `[1]`, `[2]` и списком источников | Академические / документационные задачи |
| `MULTILINGUAL` | Определяет язык вопроса, отвечает на нём | Мультиязычные базы знаний |

По умолчанию `api/app.py` использует шаблон `BASE`. Для production-сценариев рекомендуется `STRICT`.

```python
from llm.prompt_templates import get_template

tmpl = get_template("strict")
system_msg = tmpl.format_system(context="...retrieved chunks...")
user_msg   = tmpl.format_user("Что такое Python?")
```

#### `TokenBudgetManager` (`token_budget.py`)

Отвечает за усечение списка чанков под контекстное окно модели.

```python
mgr = TokenBudgetManager(model="gpt-4o-mini", reserved_tokens=1000)
safe_chunks = mgr.fit_chunks(chunks)   # наибольший префикс, влезающий в бюджет
remaining   = mgr.remaining(chunks)   # оставшийся запас токенов
```

Размеры контекстных окон (токены):

| Модель | Окно |
|---|---|
| gpt-4o, gpt-4o-mini, gpt-4-turbo | 128 000 |
| gpt-4 | 8 192 |
| gpt-3.5-turbo | 16 385 |
| claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5-20251001 | 200 000 |
| Неизвестная (Ollama и др.) | 8 000 (fallback) |

### `retrieval/`

`RAGPipeline` — основной класс пайплайна:

- `ask(question, collection)` → `RAGResponse` с полями `answer`, `sources`, `confidence`, `latency_ms`.
- `ask_stream(question, collection)` → `AsyncGenerator[str]` — потоковая генерация токен за токеном.
- `ask_multi(question, collections)` — параллельный поиск по нескольким коллекциям через `asyncio.gather`.
- `_retrieve(question, collection)` → список `SourceChunk` с полями `text`, `score`, `doc_id`, `metadata`.

Дистанция из VectorDB преобразуется в score по формуле: `score = 1 / (1 + distance)`.

Если retrieval не нашёл ни одного чанка выше порога, пайплайн возвращает константный ответ `FALLBACK_ANSWER` («К сожалению, я не нашёл информации по вашему вопросу») без вызова LLM.

#### Реранкеры (`rerankers.py`)

Опциональный шаг переранжирования результатов retrieval перед формированием промпта. Реализуют ABC `Reranker` из `retrieval/ports.py`.

| Класс | Описание |
|---|---|
| `CrossEncoderReranker` | Использует `sentence_transformers.CrossEncoder`. Нормализует скоры через sigmoid. Рекомендован при необходимости точного ранжирования по паре (вопрос, чанк). |
| `MMRReranker` | Maximal Marginal Relevance — баланс релевантности и разнообразия. Формула: `λ·sim(doc, query) − (1−λ)·max_s sim(doc, s)`. При `embed_fn=None` деградирует до сортировки по score. |

```python
from retrieval.rerankers import CrossEncoderReranker, MMRReranker

# Cross-encoder (точный, медленнее)
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(query, source_chunks)

# MMR (разнообразие результатов, быстрее)
reranker = MMRReranker(embed_fn=embed, lambda_=0.5)
reranked = reranker.rerank(query, source_chunks)
```

Реранкеры не включены в `RAGPipeline` по умолчанию — применяются как опциональный постпроцессинг.

#### TTL-кэш ответов

`RAGPipeline` содержит встроенный in-memory кэш повторяющихся вопросов.

```python
pipeline = RAGPipeline(
    ...,
    cache_ttl=300.0,   # TTL в секундах, по умолчанию 5 минут
                       # 0 — отключить кэш полностью
)
```

**Поведение:**
- Ключ кэша: `(question, collection)` — одинаковый вопрос в разных коллекциях хранится отдельно.
- При cache hit LLM и embed **не вызываются** — возвращается тот же объект `RAGResponse`.
- `ask_stream()` кэш не использует (генераторы не кэшируемы).
- Кэш живёт в памяти экземпляра `RAGPipeline` — не переживает перезапуск приложения.
- В логе при cache hit появляется запись `rag_cache_hit`; при обычном вызове — `rag_call` с `cache_hit=False`.

### `chunking/`

Модуль предоставляет три стратегии разбивки и четыре загрузчика документов.

#### Стратегии разбивки (`adapters.py`)

| Класс | Описание |
|---|---|
| `FixedSizeChunker` | Скользящее окно `RecursiveCharacterTextSplitter`. Универсальный вариант. |
| `ByHeaderChunker` | Разбивка по Markdown-заголовкам `#`–`######`. Если секция > `chunk_size`, применяет fallback на `FixedSizeChunker`. Секции короче `min_chunk_size=30` пропускаются. |
| `SemanticChunker` | Объединяет параграфы по cosine similarity их эмбеддингов (если задан `embed_fn`) или жадно по размеру. |

```python
from chunking import FixedSizeChunker, ByHeaderChunker, SemanticChunker, ingest

# Простая разбивка
chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(text)  # list[Chunk]

# По заголовкам (Markdown)
chunker = ByHeaderChunker(chunk_size=800)

# Семантическая разбивка с эмбеддингами
chunker = SemanticChunker(embed_fn=my_embed_fn, similarity_threshold=0.8)

# Высокоуровневый вход: load + chunk по расширению файла
chunks = ingest("docs/readme.md", chunker=chunker)
```

#### Загрузчики документов (`loaders.py`)

| Класс | Форматы | Примечание |
|---|---|---|
| `TextLoader` | `.txt` | UTF-8, без обработки |
| `MarkdownLoader` | `.md` | Параметр `strip_markup=False` удаляет разметку regex |
| `HTMLLoader` | `.html` | Парсинг через BeautifulSoup4 |
| `PDFLoader` | `.pdf` | Извлечение текста через pypdf |

Функция `ingest(path, chunker)` автоматически выбирает загрузчик по расширению и разбивает документ. Стабильный `id` каждого чанка генерируется из хэша текста + индекса.

#### Поддерживаемые форматы в `reindex.py`

`.txt`, `.md`, `.pdf`, `.docx`, `.html`

#### CLI (`chunking/ingest.py`)

```bash
# Разбить файл и вывести чанки в JSON
PYTHONPATH=. python chunking/ingest.py docs/readme.md --format json

# Разбить и проиндексировать в ChromaDB
PYTHONPATH=. python chunking/ingest.py docs/readme.md --index --collection my_col

# Задать параметры разбивки
PYTHONPATH=. python chunking/ingest.py docs/readme.md --chunk-size 300 --chunk-overlap 30
```

#### Низкоуровневая утилита (обратная совместимость)

```python
from chunking import chunk_text

chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)  # list[str]
```

### `api/`

#### Аутентификация (`middleware/auth.py`)

Зависимость `require_auth` поддерживает два метода:
- `X-API-Key: <key>` — прямое сравнение с `settings.api_secret_key`.
- `Authorization: Bearer <token>` — верификация JWT (алгоритм из `JWT_ALGORITHM`).

#### Точка входа: `main.py` vs `app.py`

`api/main.py` — однострочный re-export (`from api.app import app`). Нужен только как стабильный адрес для uvicorn (`uvicorn api.main:app`). Всё приложение живёт в `api/app.py`.

#### Lifespan (`app.py`)

При старте приложения в `app.state` кладутся:
- `embed_service` — `EmbeddingService`
- `pipeline` — `RAGPipeline`
- `llm`, `vector_db`, `store_client`, `store_backend`

Это позволяет роутерам получать компоненты через `Depends(get_pipeline)` без пересоздания на каждый запрос.

После инициализации запускается фоновая задача `_seed_docs_task()` — она автоматически индексирует `.md`-файлы из директории `docs/` в коллекцию `default` при старте. Файлы обрабатываются батчами по 20 штук (`_SEED_FILE_BATCH = 20`) с единым `embed_batch()` и `db.add()` на батч для минимального числа обращений к сервисам.

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

### `GET /v1/collections` — список коллекций

Возвращает список всех коллекций с числом векторов в каждой. Работает как с ChromaDB, так и с Qdrant.

```json
{
  "collections": [
    { "name": "default", "count": 342 },
    { "name": "bench_256", "count": 189 }
  ]
}
```

### `DELETE /v1/collections/{name}` — удалить коллекцию

```json
{ "deleted": "bench_256" }
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

### Структура тест-датасета (`evaluation/testcases_dataset.py`)

Фиксированный датасет из **45 тест-кейсов** по Python-теории:

| Категория | Кол-во | Назначение |
|---|---|---|
| `positive` | 30 | Вопрос, ответ на который есть в базе знаний |
| `negative` | 10 | Вопрос, ответа нет в базе — проверяет галлюцинации |
| `multi_hop` | 5 | Требует объединения информации из нескольких источников |

`TestCase` — frozen dataclass с полями: `question`, `ground_truth`, `source_doc`, `category`.

Путь к базе знаний задаётся переменной окружения `KNOWLEDGE_BASE_PATH` (default: `"Python_Theory"`).

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

## Быстрый старт (5 шагов)

Полный стек за 5 шагов — от клонирования до первого ответа:

```bash
# 1. Клонировать и войти
git clone <repo-url> rag-platform && cd rag-platform

# 2. Настроить окружение
cp .env.example .env
# Отредактировать .env: задать API_SECRET_KEY, выбрать LLM_PROVIDER

# 3. Поднять все сервисы
docker compose up --build -d

# 4. Скачать модели Ollama (если LLM_PROVIDER=ollama)
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text

# 5. Проверить что всё работает
curl http://localhost:8080/v1/health
```

После этого:
- Swagger UI: http://localhost:8080/docs
- Grafana: http://localhost:3000 (admin / admin)
- Jaeger: http://localhost:16686
- Prometheus: http://localhost:9090

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
# Отредактировать .env (файл содержит все переменные с комментариями)
```

В `.env.example` предусмотрены переменные для всех компонентов: LLM (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL`), embeddings, ChromaDB, Qdrant, API-ключи, JWT, OTel.

> **Важно:** если `CHROMA_PERSIST_DIR` задан — используется локальный `PersistentClient` (без сервера). Если пусто — `HttpClient` (требует запущенного ChromaDB-контейнера).

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
```

| Сервис | URL | Описание |
|---|---|---|
| **API** | http://localhost:8080 | FastAPI (`/docs` — Swagger UI) |
| **ChromaDB** | http://localhost:8000 | VectorDB (HTTP-режим) |
| **Qdrant** | http://localhost:6333 | Альтернативный VectorDB |
| **Ollama** | http://localhost:11434 | Локальный LLM + embeddings |
| **Prometheus** | http://localhost:9090 | Метрики |
| **Grafana** | http://localhost:3000 | Дашборды (admin / admin) |
| **Jaeger** | http://localhost:16686 | Трейсинг |

> **Dockerfile:** использует `python:3.11-slim` с CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) — предотвращает загрузку 2 ГБ CUDA-пакетов.

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

## curl-примеры

Все примеры используют `X-API-Key` аутентификацию. Подставьте значение из `.env`:

```bash
export KEY=your-api-secret-key
export BASE=http://localhost:8080/v1
```

### Здоровье системы (без аутентификации)

```bash
curl $BASE/health
```

```json
{
  "status": "ok",
  "components": {
    "vector_store": {"status": "ok", "detail": "backend=chroma, vectors=342"},
    "llm": {"status": "ok", "detail": "OllamaProvider, url=http://localhost:11434/v1"},
    "cache": {"status": "not_configured", "detail": null}
  }
}
```

### Загрузка документа — текст

```bash
curl -X POST $BASE/ingest \
     -H "X-API-Key: $KEY" \
     -F "text=Python — интерпретируемый язык программирования общего назначения." \
     -F "collection=default" \
     -F "doc_id=intro"
```

```json
{"job_id": "a3f1c2d4", "status": "pending"}
```

### Загрузка документа — файл

```bash
curl -X POST $BASE/ingest \
     -H "X-API-Key: $KEY" \
     -F "file=@README.md" \
     -F "collection=default"
```

### Загрузка документа — URL

```bash
curl -X POST $BASE/ingest \
     -H "X-API-Key: $KEY" \
     -F "url=https://docs.python.org/3/tutorial/introduction.html" \
     -F "collection=default"
```

### Статус задачи индексации

```bash
curl $BASE/ingest/a3f1c2d4 -H "X-API-Key: $KEY"
```

```json
{"job_id": "a3f1c2d4", "status": "done", "chunks_indexed": 12, "collection": "default", "error": null}
```

### Семантический поиск (лимит 500/мин)

```bash
curl -X POST $BASE/search \
     -H "X-API-Key: $KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "Что такое генератор?", "collection": "default", "n_results": 3}'
```

```json
{
  "results": [
    {"id": "intro_chunk0", "text": "Генератор — функция с yield...", "score": 0.87, "metadata": {}}
  ],
  "query": "Что такое генератор?",
  "collection": "default"
}
```

### Вопрос — полный RAG-цикл (лимит 100/мин)

```bash
curl -X POST $BASE/ask \
     -H "X-API-Key: $KEY" \
     -H "Content-Type: application/json" \
     -d '{"question": "Что такое декоратор в Python?", "collection": "default"}'
```

```json
{
  "answer": "Декоратор — это функция высшего порядка...",
  "sources": [{"text": "...", "score": 0.91, "doc_id": "intro_chunk2", "metadata": {}}],
  "confidence": 0.91,
  "latency_ms": 312.4
}
```

### Вопрос — потоковый режим (SSE)

```bash
curl -X POST $BASE/ask \
     -H "X-API-Key: $KEY" \
     -H "Content-Type: application/json" \
     -d '{"question": "Объясни генераторы", "collection": "default", "stream": true}'
```

```
data: {"token": "Генератор"}
data: {"token": " —"}
data: {"token": " функция"}
...
data: [DONE]
```

### Список коллекций

```bash
curl $BASE/collections -H "X-API-Key: $KEY"
```

```json
{"collections": [{"name": "default", "vectors_count": 342}]}
```

### Удалить коллекцию

```bash
curl -X DELETE $BASE/collections/bench_256 -H "X-API-Key: $KEY"
```

```json
{"deleted": "bench_256"}
```

### Запуск RAGAS-оценки

```bash
curl -X POST $BASE/eval/run \
     -H "X-API-Key: $KEY" \
     -H "Content-Type: application/json" \
     -d '{"mode": "mock", "max_cases": 5, "output_path": "eval_report.html"}'
```

```json
{"job_id": "e9b2c1a7", "status": "pending"}
```

### Статус оценки

```bash
curl $BASE/eval/run/e9b2c1a7 -H "X-API-Key: $KEY"
```

```json
{
  "job_id": "e9b2c1a7",
  "status": "done",
  "report_path": "eval_report.html",
  "hallucination_pass": 0,
  "hallucination_total": 10,
  "ragas_avg": {
    "faithfulness": null,
    "answer_relevancy": null,
    "context_precision": null,
    "context_recall": null
  },
  "error": null
}
```

> **Примечание:** в `mock`-режиме RAGAS-метрики равны `null` (нет реального LLM для оценки). `hallucination_pass` показывает, сколько negative-кейсов модель корректно отклонила.

### Prometheus метрики (без аутентификации)

```bash
curl $BASE/metrics
# rag_requests_total{status="success"} 5
# rag_latency_seconds_sum 1.532
# ...
```

### Заголовки трассировки

Каждый ответ содержит `X-Request-ID` — сквозной идентификатор запроса, который передаётся в OTel-спан. Можно передать свой:

```bash
curl -X POST $BASE/ask \
     -H "X-API-Key: $KEY" \
     -H "X-Request-ID: my-trace-42" \
     -H "Content-Type: application/json" \
     -d '{"question": "Python?", "collection": "default"}'
# Ответ: X-Request-ID: my-trace-42
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

## Как добавить новый LLM-провайдер

Платформа строится на паттерне Ports & Adapters. Весь код приложения зависит только от `LLMProvider` ABC — добавить нового провайдера можно без изменения пайплайна или API.

### 1. Реализовать адаптер (`llm/adapters.py`)

```python
from typing import AsyncGenerator
from .llm_dataclasses import Message
from .ports import LLMProvider


class MyProvider(LLMProvider):
    """Провайдер для MyLLM API."""

    def __init__(self, model: str = "my-model", api_key: str | None = None) -> None:
        import my_llm_sdk  # ленивый импорт — не падает при незаданном ключе

        self.model = model
        self.client = my_llm_sdk.AsyncClient(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        payload = [{"role": m.role, "content": m.content} for m in messages]

        if not stream:
            response = await self.client.chat(model=self.model, messages=payload)
            return response.text or ""

        async def _stream() -> AsyncGenerator[str, None]:
            async for chunk in await self.client.chat(
                model=self.model, messages=payload, stream=True
            ):
                if chunk.delta:
                    yield chunk.delta

        return _stream()
```

**Обязательные требования:**
- `self.model` — атрибут строкой (используется в Prometheus-метриках и OTel-спанах).
- `complete()` возвращает `str` при `stream=False` и `AsyncGenerator[str, None]` при `stream=True`.
- Нельзя возвращать `None` — используйте `""` как fallback для пустых ответов.

### 2. Зарегистрировать в фабрике (`config.py`)

```python
def build_llm_provider(settings: Settings) -> LLMProvider:
    match settings.llm_provider:
        case "openai":
            return OpenAIProvider(model=settings.llm_model, api_key=settings.openai_api_key)
        case "anthropic":
            return AnthropicProvider(model=settings.llm_model, api_key=settings.anthropic_api_key)
        case "ollama":
            return OllamaProvider(model=settings.llm_model, base_url=settings.ollama_base_url)
        case "my_provider":                           # ← добавить
            return MyProvider(model=settings.llm_model, api_key=settings.my_api_key)
        case _:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
```

### 3. Добавить переменные окружения (`.env` / `config.py`)

```bash
# .env
LLM_PROVIDER=my_provider
LLM_MODEL=my-model-v1
MY_API_KEY=key-...
```

В `config.py` в класс `Settings`:
```python
my_api_key: str | None = Field(default=None, alias="MY_API_KEY")
```

### 4. Написать тест

```python
import pytest
from unittest.mock import MagicMock, AsyncMock
from llm.llm_dataclasses import Message


@pytest.mark.asyncio
async def test_my_provider_complete_no_stream() -> None:
    provider = MyProvider.__new__(MyProvider)
    provider.model = "my-model"

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(
        return_value=MagicMock(text="Тестовый ответ")
    )
    provider.client = mock_client

    result = await provider.complete([Message(role="user", content="Привет")])
    assert result == "Тестовый ответ"
    mock_client.chat.assert_awaited_once()
```

### Чеклист

- [ ] `MyProvider` наследует `LLMProvider`, переопределяет `complete()`
- [ ] `self.model` задан строкой (нужен для метрик)
- [ ] Ленивый импорт SDK в `__init__` (не ломает старт без пакета)
- [ ] Зарегистрирован в `build_llm_provider()` в `config.py`
- [ ] `LLM_PROVIDER=my_provider` в `.env`
- [ ] Юнит-тест с мок-клиентом написан

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

---

## Chunking: стратегии и загрузчики

> Подробная документация — в разделе [Модули → `chunking/`](#chunking-1).

Краткая шпаргалка по выбору стратегии:

| Ситуация | Рекомендация |
|---|---|
| Произвольный текст, нет структуры | `FixedSizeChunker` |
| Markdown-документация с заголовками | `ByHeaderChunker` |
| Длинные параграфы, нужна семантическая граница | `SemanticChunker` с `embed_fn` |
| Нет embed-модели, нужна скорость | `SemanticChunker` без `embed_fn` (жадная разбивка) |

---

## Retrieval: переранжирование

> Подробная документация — в разделе [Модули → `retrieval/`](#retrieval).

Реранкеры применяются **после** vector search, **перед** построением промпта:

```python
# Пример использования CrossEncoderReranker вручную
from retrieval.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
pipeline = RAGPipeline(..., reranker=reranker)
```

Для MMR выставляйте `lambda_` в зависимости от задачи:
- `lambda_=1.0` — только релевантность (аналог обычного ранжирования)
- `lambda_=0.5` — баланс релевантности и разнообразия (рекомендуется)
- `lambda_=0.0` — максимальное разнообразие

---

## Коллекции API

Управление коллекциями доступно через REST API:

```bash
# Список коллекций
curl -H "X-API-Key: $API_SECRET_KEY" http://localhost:8080/v1/collections

# Удалить коллекцию
curl -X DELETE -H "X-API-Key: $API_SECRET_KEY" \
     http://localhost:8080/v1/collections/bench_256
```

Оба endpoint работают с ChromaDB и Qdrant — бэкенд определяется переменной `VECTOR_STORE_BACKEND`.
