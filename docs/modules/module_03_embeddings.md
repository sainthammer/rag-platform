# Модуль 3 — embeddings/

**Файлы:** `embeddings/ports.py`, `embeddings/adapters.py`, `embeddings/cache.py`, `embeddings/service.py`

Модуль превращает текст в числовой вектор (embedding). Построен на паттерне Ports & Adapters.

## Зачем нужны эмбеддинги

Текст нельзя напрямую сравнить математически. Вектор — можно. Модель обучена так, что похожие по смыслу тексты дают близкие векторы:

```
"как установить Python"  →  [0.12, -0.34, 0.87, ...]
"инструкция установки Python"  →  [0.11, -0.35, 0.89, ...]  ← близко!
"рецепт борща"  →  [-0.91, 0.22, -0.13, ...]                ← далеко
```

## Структура

```
embeddings/
  ports.py      абстракция EmbeddingService
  adapters.py   четыре реализации
  cache.py      SQLite-кэш эмбеддингов
  service.py    фабрика: собирает сервис + кэш
  utils.py      L2-нормализация
```

## ports.py — контракт

Весь остальной код работает только с этим интерфейсом:

```python
class EmbeddingService(ABC):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def dimension(self) -> int: ...
```

- `embed` — один текст → один вектор
- `embed_batch` — список текстов → список векторов (эффективнее API)
- `dimension` — размерность вектора (нужна при создании коллекции Qdrant)

## adapters.py — четыре реализации

### `OpenAIEmbeddingService`

Вызывает OpenAI Embeddings API. При большом батче автоматически делит на части (лимит 2048 текстов за один вызов).

```python
service = OpenAIEmbeddingService(
    model="text-embedding-3-small",
    api_key="sk-...",
    normalize=True,
    batch_limit=2048,
)
```

Известные модели и размерности:
- `text-embedding-3-small` → 1536
- `text-embedding-3-large` → 3072
- `text-embedding-ada-002` → 1536

### `SentenceTransformersService`

Запускает модель **локально** через HuggingFace. Не требует интернета и API-ключа после первой загрузки.

```python
service = SentenceTransformersService(
    model_name="BAAI/bge-m3",   # скачивается один раз
    normalize=True,
)
```

`BAAI/bge-m3` — многоязычная модель (русский, английский и ещё 100+ языков), размерность 1024.

### `OllamaEmbeddingService`

Локальные embedding-модели через Ollama. Использует тот же OpenAI-совместимый API.

```python
service = OllamaEmbeddingService(
    model="nomic-embed-text",
    base_url="http://localhost:11434/v1",
)
```

### `FakeEmbeddingService`

Только для unit-тестов. Ключевые свойства:

- **Детерминированная:** один и тот же текст → один и тот же вектор (SHA256 хэш)
- **Без сети:** никаких внешних вызовов
- **Инспектируемая:** `self.calls` хранит все вызовы — можно проверить кэш

```python
fake = FakeEmbeddingService(size=8)
vec1 = fake.embed("hello")
vec2 = fake.embed("hello")
assert vec1 == vec2          # детерминированная
print(len(fake.calls))       # 2 — сколько раз вызывали
```

## cache.py — SQLite-кэш

Кэш хранит результаты векторизации по ключу `sha256(text + model_name)`. Один и тот же текст не векторизуется дважды.

**Зачем:** каждый API-вызов к OpenAI стоит денег. При индексации большого корпуса документов один и тот же чанк может встречаться несколько раз. Кэш устраняет лишние расходы.

### Схема таблицы SQLite

```sql
CREATE TABLE embeddings (
    key          TEXT PRIMARY KEY,   -- sha256(text + model_name)
    model_name   TEXT NOT NULL,
    text_hash    TEXT NOT NULL,
    vector_json  TEXT NOT NULL,      -- вектор как JSON-массив
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
)
```

### Как работает

```
текст + имя_модели
        ↓
  sha256 → ключ
        ↓
  ищем в SQLite
        ↓
  нашли?  ──ДА──▶ возвращаем вектор из кэша
           │
           НЕТ
           ↓
  вычисляем вектор через провайдера
           ↓
  сохраняем в SQLite
           ↓
  возвращаем вектор
```

### Batch-операции

`get_many` / `set_many` — работают с несколькими текстами за одну SQL-транзакцию. При индексации документа с 50 чанками — один SELECT со всеми ключами, один INSERT для отсутствующих.

### in-memory режим

```python
cache = EmbeddingCache(path=":memory:")  # для тестов, без файла на диске
```

## service.py — сборка с кэшем

`build_embedding_service(settings)` создаёт финальный сервис:

```
settings.embedding_provider → выбор адаптера
settings.embedding_cache_enabled → обернуть в CachedEmbeddingService?
```

`CachedEmbeddingService` — декоратор над любым `EmbeddingService`. Перехватывает вызовы `embed` / `embed_batch`, проверяет кэш, вычисляет только отсутствующие векторы.

## utils.py — нормализация

**L2-нормализация** приводит длину вектора к 1 (единичный вектор):

```
v_norm = v / ||v||     где ||v|| = sqrt(sum(x² for x in v))
```

Зачем:
- Косинусное сходство нормализованных векторов = скалярное произведение (быстрее)
- Все векторы сопоставимы независимо от длины исходного текста
- ChromaDB и Qdrant лучше работают с нормализованными векторами
