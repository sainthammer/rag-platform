# Модуль `api/` — HTTP API на FastAPI

REST API для взаимодействия с RAG-платформой. Обрабатывает запросы, проверяет аутентификацию и делегирует работу компонентам из `llm/` и `vector_store/`.

---

## Структура файлов

```
api/
├── app.py              # Точка входа: FastAPI app + lifespan
├── deps.py             # Dependency injection: get_pipeline, get_vector_db, get_llm
├── schemas.py          # Pydantic-модели запросов и ответов
├── middleware/
│   └── auth.py         # Зависимость require_auth: X-API-Key + JWT Bearer
└── routers/
    ├── health.py       # GET  /v1/health
    └── collections.py  # GET  /v1/collections, DELETE /v1/collections/{name}
```

---

## Запуск

```bash
# Из корня репозитория
PYTHONPATH=. .venv/bin/python -m uvicorn api.app:app --reload

# С указанием хоста и порта
PYTHONPATH=. .venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

После старта в логах будет напечатан актуальный API-ключ:
```
INFO: API key      : dev-secret-key-change-before-deploy
```

Swagger UI доступен по адресу: `http://localhost:8000/docs`

---

## Аутентификация

Все эндпоинты кроме `GET /v1/health` требуют аутентификации.
Поддерживаются два способа — достаточно любого одного.

### X-API-Key

Передаётся в заголовке запроса. Значение должно совпадать с `API_SECRET_KEY` из `.env`.

```bash
curl http://localhost:8000/v1/collections \
  -H "X-API-Key: dev-secret-key-change-before-deploy"
```

### JWT Bearer

JWT-токен, подписанный тем же `API_SECRET_KEY`. Алгоритм подписи задаётся через `JWT_ALGORITHM` (по умолчанию `HS256`).

```bash
curl http://localhost:8000/v1/collections \
  -H "Authorization: Bearer <jwt-token>"
```

**Как сгенерировать токен:**

```python
from jose import jwt
from datetime import datetime, timedelta, UTC
from config import settings

token = jwt.encode(
    {"sub": "user-id", "exp": datetime.now(UTC) + timedelta(hours=1)},
    settings.api_secret_key,
    algorithm=settings.jwt_algorithm,
)
```

### Как устроена проверка (`middleware/auth.py`)

```
входящий запрос
       │
       ├── есть заголовок X-API-Key?
       │       └── совпадает с settings.api_secret_key? → OK
       │
       ├── есть заголовок Authorization: Bearer?
       │       └── jwt.decode() проверяет подпись + exp → OK
       │
       └── ничего не прошло → 401 Unauthorized
```

`require_auth` — обычная FastAPI-зависимость. Подключается к роутеру через `dependencies=[Depends(require_auth)]`. FastAPI автоматически добавляет схемы безопасности в OpenAPI-спецификацию, и в Swagger UI появляется кнопка **Authorize**.

### Авторизация в Swagger UI

1. Открыть `http://localhost:8000/docs`
2. Нажать **Authorize** (замок в правом верхнем углу)
3. В поле **X-API-Key** ввести значение `API_SECRET_KEY` из `.env`
4. Нажать **Authorize** → **Close**

После этого все запросы из UI будут автоматически включать заголовок.

---

## Эндпоинты

### `GET /v1/health` — без аутентификации

Проверяет состояние каждого компонента. Используется как liveness/readiness probe.

**Ответ:**
```json
{
  "status": "ok",
  "components": {
    "vector_store": { "status": "ok", "detail": "backend=chroma, vectors=42" },
    "llm":          { "status": "ok", "detail": "OllamaProvider, url=http://localhost:11434/v1" },
    "cache":        { "status": "not_configured" }
  }
}
```

Статусы компонентов: `ok` | `error` | `not_configured`.
Общий `status` = `"degraded"` только если хотя бы один компонент вернул `"error"`.
`not_configured` не считается ошибкой.

---

### `GET /v1/collections` — 🔒 требует аутентификации

Список всех коллекций векторного хранилища.

**Ответ:**
```json
{
  "collections": [
    { "name": "default", "vectors_count": 1024 },
    { "name": "docs",    "vectors_count": 512  }
  ]
}
```

---

### `DELETE /v1/collections/{name}` — 🔒 требует аутентификации

Безвозвратно удаляет коллекцию и все её векторы.

```bash
curl -X DELETE http://localhost:8000/v1/collections/docs \
  -H "X-API-Key: dev-secret-key-change-before-deploy"
```

**Ответ:**
```json
{ "deleted": "docs" }
```

**404** — если коллекция не найдена.

---

## Как работает lifespan (`app.py`)

При старте сервера FastAPI выполняет `lifespan()` один раз перед первым запросом. Все компоненты создаются здесь и кладутся в `app.state` — оттуда их забирают роутеры через зависимости из `deps.py`.

```
uvicorn запускает app
    │
    └── lifespan()
          ├── build_llm_provider()  → app.state.llm
          ├── build_vector_db()     → app.state.vector_db
          ├── vector_db.client      → app.state.store_client  (для list/delete коллекций)
          ├── build_embed_fn()      → передаётся в pipeline
          └── RAGPipeline(...)      → app.state.pipeline
```

`store_client` хранится отдельно от `vector_db` потому что `VectorDB` работает с одной конкретной коллекцией, а для операций над всеми коллекциями нужен доступ к raw-клиенту (chromadb.Client или QdrantClient).

---

## Как добавить новый эндпоинт

1. Создать файл `api/routers/my_router.py`:

```python
from fastapi import APIRouter, Depends
from api.deps import get_pipeline
from llm.pipeline import RAGPipeline

router = APIRouter(tags=["my-feature"])

@router.post("/query")
async def query(text: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    return {"answer": await pipeline.run(text)}
```

2. Подключить в `app.py`:

```python
from api.routers import my_router

app.include_router(
    my_router.router,
    prefix="/v1",
    dependencies=[Depends(require_auth)],  # если нужна авторизация
)
```

---

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `API_SECRET_KEY` | `change-me-in-production` | Ключ для X-API-Key и подписи JWT |
| `JWT_ALGORITHM` | `HS256` | Алгоритм подписи JWT (`HS256`, `HS512`, `RS256`) |
| `API_HOST` | `0.0.0.0` | Хост для uvicorn |
| `API_PORT` | `8080` | Порт для uvicorn |
