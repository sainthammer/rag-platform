# Модуль 1 — config.py

**Файл:** `config.py`

Единая точка правды для всей конфигурации. Весь проект читает настройки отсюда — никто не читает `.env` самостоятельно.

## Как работает

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

`pydantic-settings` при создании `Settings()`:
1. Читает `.env` файл
2. Читает переменные окружения (они имеют **приоритет** над `.env`)
3. Валидирует типы (строки, числа, bool) и применяет дефолты
4. `extra="ignore"` — неизвестные переменные молча игнорируются

## Что можно настраивать

### LLM — выбор модели и провайдера

| Переменная | Дефолт | Описание |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Провайдер: `openai`, `anthropic`, `ollama` |
| `LLM_MODEL` | *(по провайдеру)* | Имя модели; если пусто — подставляется дефолт |
| `OPENAI_API_KEY` | `""` | Ключ OpenAI |
| `ANTHROPIC_API_KEY` | `""` | Ключ Anthropic |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | URL Ollama-сервера |

Дефолты по провайдеру (`_PROVIDER_DEFAULTS`):
- `openai` → `gpt-4o-mini`
- `anthropic` → `claude-haiku-4-5-20251001`
- `ollama` → `llama3.2`

### Embeddings

| Переменная | Дефолт | Описание |
|---|---|---|
| `EMBEDDING_PROVIDER` | `sentence-transformers` | Провайдер: `openai`, `sentence-transformers` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Имя модели |
| `EMBEDDING_DIMENSION` | `1024` | Размерность вектора (нужна для Qdrant) |
| `EMBEDDING_NORMALIZE` | `True` | L2-нормализация векторов |
| `EMBEDDING_CACHE_ENABLED` | `True` | Включить SQLite-кэш |
| `EMBEDDING_CACHE_PATH` | `.cache/embeddings.sqlite3` | Путь к файлу кэша |

### Vector Store

| Переменная | Дефолт | Описание |
|---|---|---|
| `VECTOR_STORE_BACKEND` | `chroma` | Бэкенд: `chroma`, `qdrant` |
| `DEFAULT_COLLECTION` | `default` | Коллекция по умолчанию |
| `CHROMA_PERSIST_DIR` | `""` | Если задан — Chroma хранит данные в файле (без Docker) |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost:8000` | Chroma через HTTP |
| `QDRANT_URL` | `http://localhost:6333` | URL Qdrant |
| `QDRANT_API_KEY` | `""` | API-ключ Qdrant (для облачного деплоя) |

### API

| Переменная | Дефолт | Описание |
|---|---|---|
| `API_SECRET_KEY` | `change-me-in-production` | **Обязательно менять в prod!** |
| `JWT_ALGORITHM` | `HS256` | Алгоритм подписи JWT-токенов |
| `API_HOST` / `API_PORT` | `0.0.0.0:8080` | Адрес сервера |

### Observability

| Переменная | Дефолт | Описание |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Endpoint для трейсов (Jaeger) |

## Фабрики — главная идея

Вместо того чтобы везде писать `if provider == "openai": ...`, это делается один раз в `config.py`. Фабрики читают настройки и создают нужную реализацию:

```python
# Создаёт OpenAIProvider / AnthropicProvider / OllamaProvider
llm = build_llm_provider(settings)

# Создаёт ChromaDB / QdrantVectorStore (с BM25)
db = build_vector_db(settings, collection="my-docs")

# Создаёт EmbeddingService + опционально оборачивает в CachedEmbeddingService
emb = build_embedding_service(settings)
```

Остальной код работает с абстракциями (`LLMProvider`, `VectorDB`, `EmbeddingService`) и не знает, какой именно провайдер используется.

## Синглтон

```python
# Создаётся один раз при импорте модуля
settings = Settings()
```

Весь проект импортирует `from config import settings` — это всегда один и тот же объект. В `api/app.py` на старте сервера компоненты создаются один раз и кладутся в `app.state`.

## Пример .env файла

```env
# LLM
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Embeddings — локальная модель (без интернета)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_CACHE_ENABLED=true

# Vector Store — Chroma без Docker
VECTOR_STORE_BACKEND=chroma
CHROMA_PERSIST_DIR=./chroma_data

# API
API_SECRET_KEY=my-super-secret-key
```
