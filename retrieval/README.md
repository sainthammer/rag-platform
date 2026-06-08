# Модуль `retrieval/` — RAG-пайплайн второго поколения

Модуль отвечает за полный RAG-цикл: поиск релевантных чанков в векторном хранилище, формирование контекста и генерацию ответа через LLM.

В отличие от базового `llm/pipeline.py`, этот пайплайн возвращает структурированный ответ с источниками, уровнем уверенности и задержкой — всё что нужно для построения продакшн-API и оценки качества через RAGAS.

---

## Структура файлов

```
retrieval/
├── pipeline.py   # RAGPipeline, RAGResponse, SourceChunk
├── example.py    # Smoke-проверка без внешних сервисов
└── __init__.py   # re-export публичного API
```

---

## Публичное API

```python
from retrieval import RAGPipeline, RAGResponse, SourceChunk, FALLBACK_ANSWER
```

---

## Датаклассы

### `SourceChunk`

Один найденный фрагмент документа:

```python
@dataclass
class SourceChunk:
    text: str        # текст чанка
    score: float     # релевантность [0, 1], выше — лучше
    doc_id: str      # id документа в векторном хранилище
    metadata: dict   # произвольные метаданные: source, page, chunk_index, …
```

`score` вычисляется из дистанции ChromaDB/Qdrant по формуле `1 / (1 + distance)`:
- дистанция 0 → score 1.0 (идентичные векторы)
- дистанция 1 → score 0.5
- дистанция → ∞ → score → 0

### `RAGResponse`

Полный ответ пайплайна:

```python
@dataclass
class RAGResponse:
    answer: str                  # текст ответа от LLM или fallback-сообщение
    sources: list[SourceChunk]   # чанки, которые попали в промпт
    confidence: float            # max score среди sources, [0, 1]
    latency_ms: float            # время всего цикла в миллисекундах
```

---

## `RAGPipeline`

### Конструктор

```python
RAGPipeline(
    llm: LLMProvider,
    vector_db_factory: Callable[[str], VectorDB],
    embed_fn: Callable[[str], list[float]],
    template: PromptTemplate = BASE,
    n_results: int = 5,
    budget: TokenBudgetManager | None = None,
    score_threshold: float = 0.0,
    fallback_answer: str = FALLBACK_ANSWER,
)
```

| Параметр | Назначение |
|---|---|
| `llm` | Провайдер LLM (`OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`) |
| `vector_db_factory` | Фабрика `collection_name → VectorDB`. Вызывается на каждый `ask()` — позволяет динамически выбирать коллекцию |
| `embed_fn` | Функция векторизации запроса. Принимает строку, возвращает `list[float]` |
| `template` | Шаблон промпта из `llm/prompt_templates.py`. По умолчанию `BASE` |
| `n_results` | Сколько ближайших чанков запрашивать у VectorDB |
| `budget` | Менеджер токенного бюджета. Если не задан — создаётся с дефолтными параметрами |
| `score_threshold` | Минимальный score лучшего чанка для передачи контекста в LLM. Ниже порога — fallback |
| `fallback_answer` | Текст ответа при отсутствии релевантного контекста |

### Методы

#### `ask(question, collection, stream=False) → RAGResponse`

Полный RAG-цикл. Возвращает структурированный ответ всегда — в том числе при fallback.

При `stream=True` — LLM вызывается в потоковом режиме, но ответ накапливается в строку. `RAGResponse.answer` всегда полный. Используйте `ask_stream()` если нужны токены по одному.

```python
response = await pipeline.ask("Что такое RAG?", collection="docs")

print(response.answer)        # текст ответа
print(response.confidence)    # 0.796
print(response.latency_ms)    # 14.2

for source in response.sources:
    print(source.score, source.metadata["source"], source.text[:80])
```

#### `ask_stream(question, collection) → AsyncGenerator[str, None]`

Стриминг ответа токен за токеном. При fallback выдаёт сообщение одним блоком.

Не нужно `await` при вызове — это async-генератор:

```python
async for token in pipeline.ask_stream("Что такое RAG?", "docs"):
    print(token, end="", flush=True)
```

#### `ask_multi(question, collections) → list[RAGResponse]`

Параллельный запрос нескольких коллекций через `asyncio.gather`. Порядок результатов совпадает с порядком входного списка.

```python
responses = await pipeline.ask_multi(
    "Что такое RAG?",
    collections=["docs-ru", "docs-en", "faq"],
)
# responses[0] — из docs-ru
# responses[1] — из docs-en
# responses[2] — из faq
```

---

## Поток данных

```
question (str)
  │
  ├─▶ embed_fn(question)              → вектор запроса
  ├─▶ vector_db_factory(collection)   → VectorDB нужной коллекции
  ├─▶ db.search(embedding, n_results) → SearchResult
  │
  ├─▶ _distance_to_score(dist)        → score для каждого чанка
  ├─▶ confidence = max(scores)
  │
  ├─▶ [FALLBACK] нет чанков ИЛИ confidence < threshold
  │       └─▶ RAGResponse(fallback_answer, sources=[], …)
  │
  ├─▶ budget.fit_chunks(texts)        → чанки в рамках токенного бюджета
  ├─▶ template.format_system(context) → системный промпт
  ├─▶ template.format_user(question)  → сообщение пользователя
  └─▶ llm.complete(messages)          → answer → RAGResponse
```

---

## Fallback

Пайплайн не вызывает LLM и возвращает `fallback_answer` в двух случаях:

1. **Нет чанков** — коллекция пустая или запрос не нашёл совпадений.
2. **`confidence < score_threshold`** — чанки найдены, но все слишком далеки от вопроса семантически.

Без fallback LLM получил бы пустой контекст и начал бы генерировать ответ из своих весов — это галлюцинация. Fallback экономит токены и даёт пользователю честный ответ.

```python
# Никогда не fallback (дефолт — доверяем любому результату)
pipeline = RAGPipeline(..., score_threshold=0.0)

# Fallback если лучший чанк набрал меньше 0.4
pipeline = RAGPipeline(..., score_threshold=0.4)

# Своё сообщение
pipeline = RAGPipeline(..., fallback_answer="Информация не найдена.")
```

---

## Конфигурация через `build_embedding_service` + фабрика

Типичная инициализация в продакшн-коде:

```python
from config import settings
from embeddings import build_embedding_service
from retrieval import RAGPipeline
from vector_store.adapters import ChromaDB
from llm.adapters import OpenAIProvider

embed_service = build_embedding_service()  # читает EMBEDDING_* из .env

pipeline = RAGPipeline(
    llm=OpenAIProvider(model="gpt-4o-mini"),
    vector_db_factory=lambda name: ChromaDB(
        collection=name,
        host=settings.chroma_host,
        port=settings.chroma_port,
    ),
    embed_fn=embed_service.embed,
    score_threshold=0.3,
    n_results=5,
)
```

---

## Тесты

Интеграционные тесты используют реальный ChromaDB и mock LLM:

```bash
# Только интеграционные тесты retrieval
PYTHONPATH=. .venv/bin/python -m pytest tests/integration/test_rag_pipeline.py -v

# Все тесты проекта
PYTHONPATH=. .venv/bin/python -m pytest -v
```

Что проверяется:

| Тест | Проверка |
|---|---|
| `test_ask_sources_contain_indexed_documents` | `sources` содержит только тексты из индекса ChromaDB |
| `test_ask_sources_have_correct_metadata` | `metadata` в sources совпадает с переданным при индексации |
| `test_ask_confidence_equals_max_source_score` | `confidence` — это максимальный score среди sources |
| `test_ask_llm_receives_context_with_retrieved_texts` | системный промпт содержит найденные документы |
| `test_ask_fallback_on_empty_collection` | пустая коллекция → fallback, LLM не вызывается |
| `test_ask_fallback_when_score_below_threshold` | impossibly high threshold → fallback |
| `test_ask_stream_yields_tokens` | `ask_stream` отдаёт токены по одному |
| `test_ask_multi_all_collections_get_answer` | `ask_multi` вызывает LLM по разу на каждую коллекцию |

---

## Smoke-проверка без внешних сервисов

```bash
PYTHONPATH=. .venv/bin/python -m retrieval.example
```

Запускает полный цикл: `ask()`, fallback, `ask_stream()` и `ask_multi()` — всё с fake-embedding и fake-LLM, без API-ключей и сетевых вызовов.

---

## Отличие от `llm/pipeline.py`

| | `llm.RAGPipeline` | `retrieval.RAGPipeline` |
|---|---|---|
| Возвращает | `str \| AsyncGenerator` | `RAGResponse` |
| Источники | ✗ | ✓ `sources: list[SourceChunk]` |
| Confidence | ✗ | ✓ |
| Latency | ✗ | ✓ |
| Fallback | ✗ | ✓ |
| Несколько коллекций | ✗ | ✓ `ask_multi()` |
| Принимает коллекцию | ✗ (один `VectorDB`) | ✓ `vector_db_factory` |
