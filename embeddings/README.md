# Модуль `embeddings/` — сервис эмбеддингов и SQLite-кэш

Модуль отвечает за построение embedding-векторов для документов и пользовательских запросов.
Код следует подходу Ports & Adapters: остальная система работает через абстрактный
`EmbeddingService`, а конкретный провайдер выбирается настройками.

Сейчас реализованы:

- абстрактный `EmbeddingService`;
- `OpenAIEmbeddingService` для OpenAI Embeddings API;
- `SentenceTransformersService` для локальных моделей, включая `BAAI/bge-m3`;
- L2-нормализация по флагу `normalize`;
- SQLite-кэш с ключом `sha256(text + model_name)`;
- `CachedEmbeddingService` для прозрачного кэширования;
- `FakeEmbeddingService` для unit-тестов без внешних сервисов.

---

## Структура файлов

```
embeddings/
├── ports.py      # Абстрактный интерфейс EmbeddingService
├── adapters.py   # OpenAI, SentenceTransformers и Fake реализации
├── cache.py      # SQLite-кэш и декоратор @cached
├── service.py    # CachedEmbeddingService и фабрика build_embedding_service()
├── utils.py      # L2-нормализация и вспомогательные функции
├── __init__.py   # re-export публичного API модуля
└── README.md
```

---

## Публичное API модуля

Удобный импорт:

```python
from embeddings import (
    EmbeddingService,
    OpenAIEmbeddingService,
    SentenceTransformersService,
    CachedEmbeddingService,
    EmbeddingCache,
    build_embedding_service,
)
```

---

## Контракт `EmbeddingService`

Любая реализация должна поддерживать три метода:

```python
service.embed("text")          # один текст -> list[float]
service.embed_batch(["a", "b"]) # список текстов -> list[list[float]]
service.dimension()            # размерность вектора
```

`embed_batch()` сохраняет порядок входных текстов. Это важно для дальнейшей записи
в vector store: `ids`, `documents`, `metadatas` и `embeddings` должны совпадать по индексам.

---

## Конфигурация

Настройки читаются через `config.py` из переменных окружения:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_NORMALIZE=true
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_PATH=.cache/embeddings.sqlite3
```

Доступные значения `EMBEDDING_PROVIDER`:

- `openai` — OpenAI Embeddings API;
- `sentence-transformers` — локальная модель через пакет `sentence-transformers`.

---

## Кэширование

`EmbeddingCache` хранит векторы в SQLite. Ключ строится как:

```python
sha256(text + model_name)
```

Это означает, что один и тот же текст для разных моделей будет сохранён разными записями.
Такой формат безопасен при смене embedding-модели или размерности коллекции.

---

## Тесты

Unit-тесты модуля находятся в двух файлах:

- `tests/unit/test_embeddings.py` — проверяет `dimension()`, L2-нормализацию
  и соответствие `embed_batch()` одиночным вызовам `embed()`;
- `tests/unit/test_embedding_cache.py` — проверяет ключ кэша, запись/чтение SQLite,
  cache hit и то, что batch-вызов считает только отсутствующие тексты.

Запуск только тестов модуля embeddings:

```bash
python -m pytest tests/unit/test_embeddings.py tests/unit/test_embedding_cache.py -q
```

Запуск отдельных групп:

```bash
# Базовый контракт EmbeddingService
python -m pytest tests/unit/test_embeddings.py -q

# SQLite-кэш и CachedEmbeddingService
python -m pytest tests/unit/test_embedding_cache.py -q
```

Проверка стиля и статических правил:

```bash
python -m ruff check embeddings tests/unit/test_embeddings.py tests/unit/test_embedding_cache.py
```

---

## Пример использования

```python
from embeddings import build_embedding_service

service = build_embedding_service()

query_vector = service.embed("Что такое RAG?")
document_vectors = service.embed_batch([
    "RAG объединяет поиск и генерацию.",
    "Vector store хранит embedding-векторы чанков.",
])
```

Локальная smoke-проверка без внешних сервисов. Это то же самое, что режим `fake`:

```bash
python -m embeddings.example
```

Проверка конкретного провайдера:

```bash
python -m embeddings.example fake
python -m embeddings.example openai
python -m embeddings.example sentence-transformers
```

Что проверяет пример:

- `fake` — контракт сервиса, batch-вызов, L2-нормализацию и SQLite-кэш без сети;
- `openai` — реальный вызов OpenAI Embeddings API, требует `OPENAI_API_KEY`;
- `sentence-transformers` — локальную модель, например `BAAI/bge-m3`.

В выводе примера есть не весь embedding-вектор, а диагностические значения:

- размерность вектора;
- L2-норма первого вектора;
- первые 8 координат первого вектора;
- cosine similarity между текстами.

Полный embedding обычно не выводится: для `BAAI/bge-m3` это 1024 числа,
для `text-embedding-3-small` — 1536 чисел. Для проверки полезнее смотреть
размерность, норму и сходство между текстами.

`sentence-transformers` может скачать модель при первом запуске.
Если модель уже скачана, но HuggingFace недоступен, можно запустить в офлайн-режиме:

```bash
HF_HUB_OFFLINE=1 python -m embeddings.example sentence-transformers
```

---

## Зависимости

Основные зависимости:

- `openai` — для `OpenAIEmbeddingService`;
- `sentence-transformers` — для `SentenceTransformersService`;
- `sqlite3` — стандартная библиотека Python, используется для кэша.

Unit-тесты используют `FakeEmbeddingService`, поэтому не требуют API-ключей и сетевого доступа.
