# Модуль 5 — retrieval/

**Файлы:** `retrieval/pipeline.py`, `retrieval/retriever.py`, `retrieval/ports.py`, `retrieval/rerankers.py`

Главный «мозг» поиска. Объединяет embeddings, vector_store и llm в единый пайплайн.

## Структура

```
retrieval/
  pipeline.py   RAGPipeline — полный RAG-цикл (retrieve + generate)
  retriever.py  Retriever   — только поиск (без генерации)
  ports.py      Reranker    — абстракция реранкера
  rerankers.py  CrossEncoderReranker, MMRReranker
```

## retriever.py — компонент поиска

`Retriever` отвечает только за поиск — без LLM, без генерации. Это позволяет использовать его независимо (для оценки retrieval качества, в MCP-инструментах).

```python
retriever = Retriever(
    embed_fn=embed_service.embed,
    vector_db_factory=lambda col: build_vector_db(settings, collection=col),
    reranker=CrossEncoderReranker(),   # опционально
    fetch_k=None,                      # None → top_k * 3 при наличии reranker
)

chunks = retriever.retrieve(
    query="Как настроить авторизацию?",
    collection="docs",
    top_k=5,
    filters={"source": "manual.pdf"},  # фильтр по metadata
)
```

### Поток данных `retrieve()`

```
query (str)
  │
  ├── embed_fn(query)              → вектор запроса
  │
  ├── vector_db_factory(collection) → VectorDB
  │
  ├── db.search(vector, n=fetch_k)  → raw SearchResult
  │
  ├── → список SourceChunk (distance → score: 1/(1+d))
  │
  ├── filters?  → фильтрация по metadata (post-filter)
  │
  ├── reranker? → reranker.rerank(query, candidates)
  │
  └── candidates[:top_k]           → List[SourceChunk]
```

**fetch_k** — сколько кандидатов запросить из БД перед реранкингом. По умолчанию `top_k * 3` — берём больше, потом реранкер отбирает лучшие. Без реранкера = `top_k`.

**Фильтрация по metadata** — post-filter после векторного поиска:
```python
filters={"source": "manual.pdf"}
# оставляем только чанки, у которых metadata["source"] == "manual.pdf"
```

## pipeline.py — RAGPipeline

Полный RAG-цикл: retrieval + augmentation + generation.

### Конструктор

```python
pipeline = RAGPipeline(
    llm=build_llm_provider(settings),
    vector_db_factory=lambda name: build_vector_db(settings, collection=name),
    embed_fn=embed_service.embed,
    template=BASE,          # шаблон промпта
    n_results=5,            # сколько чанков искать
    budget=TokenBudgetManager(),
    score_threshold=0.0,    # мин. порог релевантности
    fallback_answer="...",  # ответ если нет релевантных чанков
)
```

### Три метода

**`ask(question, collection)`** — полный цикл, возвращает `RAGResponse`:

```
question
  │
  ├── _retrieve(question, collection)
  │     → (sources: list[SourceChunk], confidence: float)
  │
  ├── confidence < threshold? → вернуть fallback_answer
  │
  ├── _build_messages(question, sources)
  │     → [Message(role="system", ...), Message(role="user", ...)]
  │
  └── llm.complete(messages)
        → RAGResponse(answer, sources, confidence, latency_ms)
```

**`ask_stream(question, collection)`** — async-генератор токенов:

```python
async for token in pipeline.ask_stream("вопрос", "docs"):
    print(token, end="", flush=True)
```

Позволяет показывать ответ пользователю "в реальном времени" — каждый токен отправляется сразу по мере генерации LLM.

**`ask_multi(question, collections)`** — параллельный запрос к нескольким коллекциям:

```python
responses = await pipeline.ask_multi("вопрос", ["docs", "faq", "api-reference"])
# asyncio.gather — все три запроса летят одновременно
```

### `SourceChunk` и `RAGResponse`

```python
@dataclass
class SourceChunk:
    text: str
    score: float     # [0, 1]: 1 = идеальное совпадение
    doc_id: str
    metadata: dict

@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceChunk]   # найденные чанки с оценками
    confidence: float            # max score среди чанков
    latency_ms: float            # полное время выполнения
```

### Преобразование distance → score

```python
def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + distance)
```

- distance=0 → score=1.0 (идентичные векторы)
- distance=1 → score=0.5
- distance=9 → score=0.1

### Промпт и токенный бюджет

```python
def _build_messages(question, sources):
    chunks = self.budget.fit_chunks([s.text for s in sources])  # обрезка по токенам
    context = "\n\n---\n\n".join(chunks)
    return [
        Message(role="system", content=template.format_system(context)),
        Message(role="user", content=template.format_user(question)),
    ]
```

## ports.py — абстракция Reranker

```python
class Reranker(ABC):
    def rerank(self, query: str, candidates: list[SourceChunk]) -> list[SourceChunk]: ...
```

Принимает список кандидатов от векторного поиска, возвращает пересортированный список.

## rerankers.py — два реранкера

### `CrossEncoderReranker`

**Bi-encoder** (обычный embedding) кодирует вопрос и документ **раздельно**, потом считает cosine similarity. Быстро, но менее точно.

**Cross-encoder** смотрит на пару `(вопрос, документ)` **вместе** — это точнее, но медленнее. Используется для финального ранжирования топ-N кандидатов.

```
Bi-encoder:    embed(query)  ·  embed(doc)  =  cos_sim   ← для поиска в БД
Cross-encoder: model(query ++ doc)          =  score     ← для реранкинга
```

```python
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # лёгкая модель
)
```

Модель загружается лениво при первом вызове `rerank()`. Скоры нормализуются через sigmoid → [0, 1].

Паттерн использования: берём из БД 15 кандидатов → CrossEncoder пересортировывает → берём топ-5.

### `MMRReranker` (Maximal Marginal Relevance)

Стандартный поиск может вернуть 5 чанков об одном и том же. MMR решает эту проблему: итеративно выбирает следующий чанк, который **и релевантен, и не похож** на уже выбранные.

```
score = λ · sim(doc, query) − (1 − λ) · max(sim(doc, already_selected))
```

- `λ=1.0` → только релевантность (как без MMR)
- `λ=0.0` → только разнообразие
- `λ=0.5` → баланс (рекомендуется)

```python
reranker = MMRReranker(
    lambda_=0.5,
    embed_fn=embed_service.embed,   # нужен для вычисления doc-doc сходства
)
```

Без `embed_fn` — деградирует до сортировки по исходному score (diversity отключается).

## Фабрика `build_rag_pipeline`

```python
pipeline = build_rag_pipeline(settings, score_threshold=0.3, n_results=10)
```

Читает `EMBEDDING_PROVIDER`, `LLM_PROVIDER`, `VECTOR_STORE_BACKEND` из настроек и возвращает готовый пайплайн. Любые параметры `RAGPipeline` можно переопределить через `**kwargs`.
