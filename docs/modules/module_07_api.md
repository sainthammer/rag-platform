# Модуль 7 — api/

**Файлы:** `api/app.py`, `api/main.py`, `api/schemas.py`, `api/deps.py`, `api/jobs.py`, `api/limiter.py`, `api/middleware/`, `api/routers/`

FastAPI-сервер: точка входа для всех HTTP-запросов к RAG-платформе.

## Структура

```
api/
  app.py              FastAPI + lifespan + подключение роутеров
  main.py             точка запуска (uvicorn api.main:app)
  schemas.py          Pydantic-схемы запросов и ответов
  deps.py             FastAPI-зависимости (get_pipeline, get_embed_service...)
  jobs.py             in-memory хранилище фоновых задач
  limiter.py          rate limiting (slowapi)
  middleware/
    auth.py           X-API-Key + JWT Bearer
    request_id.py     генерация X-Request-ID
  routers/
    health.py         GET /v1/health
    ingest.py         POST /v1/ingest, GET /v1/ingest/{job_id}
    search.py         POST /v1/search
    ask.py            POST /v1/ask
    eval.py           POST /v1/eval, GET /v1/eval/{job_id}
    collections.py    GET/DELETE /v1/collections
    metrics.py        GET /v1/metrics
```

## app.py — точка сборки

### Lifespan — инициализация при старте

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Создаём все компоненты один раз при старте
    llm = build_llm_provider(settings)
    embed_service = build_embedding_service(settings)
    vector_db = build_vector_db(settings)
    pipeline = RAGPipeline(llm=llm, ...)

    # Кладём в app.state — доступны через Depends во всех роутерах
    app.state.pipeline = pipeline
    app.state.embed_service = embed_service
    app.state.vector_db = vector_db
    yield
    # Здесь можно закрыть соединения при остановке
```

### Роутеры и аутентификация

```python
# Без аутентификации
app.include_router(health.router, prefix="/v1")
app.include_router(metrics.router, prefix="/v1")   # Prometheus должен скрейпить свободно

# С аутентификацией
_auth = [Depends(require_auth)]
app.include_router(ingest.router, prefix="/v1", dependencies=_auth)
app.include_router(search.router, prefix="/v1", dependencies=_auth)
app.include_router(ask.router, prefix="/v1", dependencies=_auth)
app.include_router(eval.router, prefix="/v1", dependencies=_auth)
app.include_router(collections.router, prefix="/v1", dependencies=_auth)
```

## middleware/auth.py — аутентификация

Два способа передачи учётных данных:

**1. X-API-Key** — прямой ключ:
```
X-API-Key: my-secret-key
```

**2. JWT Bearer** — JWT-токен, подписанный тем же `api_secret_key`:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Логика проверки:
```
X-API-Key совпадает с settings.api_secret_key? → OK
иначе Bearer токен → jwt.decode(token, secret_key, algorithms=[jwt_algorithm]) → OK
иначе → 401 Unauthorized
```

`jwt.decode` автоматически проверяет подпись и срок действия (`exp`).

## middleware/request_id.py — X-Request-ID

Каждый запрос получает уникальный идентификатор:

```
1. Клиент прислал X-Request-ID?  → используем его
2. Иначе                          → генерируем UUID4

Сохраняется в request.state.request_id
Прокидывается в OpenTelemetry span → атрибут http.request_id
Возвращается клиенту в заголовке ответа X-Request-ID
```

Это позволяет найти конкретный запрос в Jaeger-трейсах зная его ID. Полезно при дебаггинге: пользователь жалуется на ошибку → смотришь его X-Request-ID в логах.

## limiter.py — rate limiting

```python
limiter = Limiter(key_func=get_remote_address)
```

Ограничение числа запросов по IP-адресу. Подключается к роутерам декоратором:
```python
@limiter.limit("10/minute")
async def ask(...): ...
```

## schemas.py — Pydantic-модели

Pydantic автоматически валидирует входные данные и генерирует OpenAPI документацию (доступна на `/docs`).

### Основные схемы

**Ingest:**
```python
class IngestJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]

class IngestStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    chunks_indexed: int = 0
    collection: str = ""
    error: str | None = None
```

**Search:**
```python
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    collection: str = "default"
    n_results: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
```

**Ask:**
```python
class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    collection: str = "default"
    stream: bool = False
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    n_results: int = Field(default=5, ge=1, le=50)

class AskResponse(BaseModel):
    answer: str
    sources: list[AskSourceItem]
    confidence: float
    latency_ms: float
```

## routers/ingest.py — асинхронная индексация

`POST /v1/ingest` принимает документ (файл / URL / текст), сразу отвечает `202 Accepted`, индексация идёт в фоне:

```
POST /v1/ingest  →  202 Accepted, job_id="abc-123"  (мгновенно)
        ↓
[asyncio.create_task(_ingest_task(...))]
        ↓ (фоновая задача)
chunk_text() → embed_batch() → db.add()    (занимает время)
        ↓
GET /v1/ingest/abc-123  →  {"status": "done", "chunks_indexed": 42}
```

**jobs.py** — простое in-memory хранилище задач. `Job` датакласс с полями `job_id`, `status`, `result`, `error`. При перезапуске сервера задачи теряются.

Источник документа — ровно один из: `file`, `url`, `text`. При `url` — загружается через `httpx.AsyncClient` с таймаутом 15 секунд.

## routers/ask.py — два режима ответа

```python
if body.stream:
    # SSE: клиент получает токены по мере генерации
    return StreamingResponse(
        _sse_generator(pipeline, question, collection),
        media_type="text/event-stream",
    )
else:
    response = await pipeline.ask(body.question, body.collection)
    return AskResponse(answer=..., sources=..., confidence=..., latency_ms=...)
```

**SSE-формат:**
```
data: {"token": "Привет"}\n\n
data: {"token": ", вот"}\n\n
data: {"token": " ответ"}\n\n
data: [DONE]\n\n
```

`Cache-Control: no-cache` и `X-Accel-Buffering: no` — отключают буферизацию на уровне nginx/proxy.

## Все эндпоинты

| Метод | Путь | Auth | Описание |
|---|---|---|---|
| GET | `/v1/health` | нет | Проверка состояния компонентов |
| GET | `/v1/metrics` | нет | Prometheus метрики |
| POST | `/v1/ingest` | да | Загрузить документ (async) |
| GET | `/v1/ingest/{job_id}` | да | Статус индексации |
| POST | `/v1/search` | да | Семантический поиск |
| POST | `/v1/ask` | да | Полный RAG-цикл |
| POST | `/v1/eval` | да | Запустить RAGAS оценку |
| GET | `/v1/eval/{job_id}` | да | Статус и результат оценки |
| GET | `/v1/collections` | да | Список коллекций |
| DELETE | `/v1/collections/{name}` | да | Удалить коллекцию |

Swagger UI доступен на `http://localhost:8080/docs` — можно тестировать все эндпоинты прямо из браузера.
