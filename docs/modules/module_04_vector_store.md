# Модуль 4 — vector_store/

**Файлы:** `vector_store/ports.py`, `vector_store/adapters.py`, `vector_store/bm25.py`, `vector_store/store_dataclasses.py`

Модуль отвечает за хранение и поиск векторов. Поддерживает два типа поиска: семантический (dense) и полнотекстовый (BM25 sparse), и их гибридную комбинацию.

## Структура

```
vector_store/
  ports.py             абстракция VectorDB
  adapters.py          ChromaDB, QdrantVectorStore, HybridVectorStore
  bm25.py              BM25SparseVectorizer
  store_dataclasses.py SearchResult, CollectionStats
  utils.py             to_uuid()
```

## ports.py — контракт VectorDB

```python
class VectorDB(ABC):
    def add(ids, embeddings, documents, metadatas): ...   # записать
    def search(query_embedding, n_results): ...           # найти похожие
    def delete(ids): ...                                  # удалить
    def count(): ...                                      # сколько записей
    def get_stats(): ...                                  # CollectionStats
```

`get_stats()` — новый метод: возвращает `CollectionStats` с числом векторов, размером на диске и в RAM (для Qdrant).

## store_dataclasses.py — структуры данных

### `SearchResult`

```python
@dataclass
class SearchResult:
    ids: list[str]          # идентификаторы найденных чанков
    documents: list[str]    # тексты чанков
    distances: list[float]  # расстояние (меньше = более похожее)
    metadatas: list[dict]   # метаданные (source, chunk_index...)
```

### `CollectionStats`

```python
@dataclass
class CollectionStats:
    collection: str
    vectors_count: int
    segments_count: int | None = None
    disk_size_bytes: int | None = None
    ram_size_bytes: int | None = None
```

## adapters.py — три реализации

### `ChromaDB`

```python
ChromaDB(collection, persist_directory=None, host, port)
```

Два режима:
```python
# Локальный файл (разработка без Docker)
persist_directory="./chroma_data"  →  PersistentClient

# HTTP-сервер (production в Docker)
persist_directory=None             →  HttpClient(host, port)
```

Все методы обёрнуты в `@retry` (tenacity): при `ConnectionError` или `TimeoutError` — до 3 попыток с экспоненциальным ожиданием (1s → 2s → 4s).

### `QdrantVectorStore`

Хранит **два вектора** для каждого документа:

```
"dense"   — плотный вектор от embedding-модели (семантика)
"sparse"  — разреженный BM25-вектор (ключевые слова)
```

```python
store = QdrantVectorStore(
    collection="docs",
    vector_size=1024,    # размерность dense-вектора
    vectorizer=BM25SparseVectorizer(),
    in_memory=True,      # для тестов
)
```

При первом вызове `add()` — если BM25-векторизатор не обучен — он автоматически обучается на загружаемых документах (`vectorizer.fit(docs)`).

Методы поиска:
- `search(query_embedding)` — только dense (семантический)
- `sparse_search(query_text)` — только BM25 (полнотекстовый)

### `HybridVectorStore`

Наследует `QdrantVectorStore`, добавляет `hybrid_search()`:

```python
results = store.hybrid_search(
    query_text="ключевые слова",
    query_embedding=[0.12, -0.34, ...],
    n_results=5,
    rrf_k=60,
    fetch_k=10,     # берём 10 от каждого поиска, потом сливаем
)
```

Внутри: dense search + sparse search → RRF fusion.

## bm25.py — как работает BM25

**BM25 (Best Match 25)** — алгоритм полнотекстового поиска (как в поисковиках). В отличие от семантического поиска находит по точным словам.

### Зачем BM25 + семантика?

| Тип поиска | Находит | Не находит |
|---|---|---|
| Семантический | «как установить Python» → «инструкция установки» | редкие термины, аббревиатуры, имена |
| BM25 | «API_KEY» → «API_KEY», «FastAPI» → «FastAPI» | синонимы, парафразы |
| **Гибридный** | **всё из обоих** | |

### Обучение и трансформация

```python
vectorizer = BM25SparseVectorizer(k1=1.5, b=0.75)
vectorizer.fit(corpus)          # строит словарь, вычисляет IDF по корпусу
sv = vectorizer.transform(text) # → SparseVector(indices=[...], values=[...])
```

`SparseVector` — только термы из текста с их BM25-весами. Для документа «Python и FastAPI» это ~5 индексов из словаря тысяч слов.

### Параметры BM25

- `k1=1.5` — насыщение частоты термина (TF saturation). Чем больше — тем важнее частота повторения
- `b=0.75` — нормализация по длине документа. `b=1` — полная нормализация, `b=0` — без неё

### Формула BM25

```
score(d, q) = Σ IDF(t) · TF(t, d) · (k1 + 1) / (TF(t, d) + k1 · (1 - b + b · |d| / avgdl))

IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

- `IDF` — обратная частота документа: редкие слова важнее
- `TF` — частота термина в документе: частые повторения важнее (но с насыщением)
- `|d| / avgdl` — нормализация по длине документа

## RRF (Reciprocal Rank Fusion)

Алгоритм слияния двух ранжированных списков. Каждый документ получает очки от обоих поисков:

```
score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
```

Документ, который высоко в обоих списках, получает наибольший итоговый score:

```
Dense:  [A, B, C, D]
Sparse: [B, C, A, E]
         ↓
RRF:    [B, A, C, D, E]    B и A — высоко в обоих
```

`k=60` — константа, сглаживающая разницу между рангами.

## utils.py — to_uuid

ChromaDB и Qdrant требуют UUID в качестве идентификаторов. `to_uuid` стабильно преобразует строку в UUID5:

```python
to_uuid("my-doc_chunk0")  # → "a3f1c2d4-..." (всегда одинаковый UUID для одной строки)
```

Это важно для `upsert`: повторная индексация одного документа обновляет те же записи, а не создаёт дубликаты.

## Выбор бэкенда

| | ChromaDB | QdrantVectorStore | HybridVectorStore |
|---|---|---|---|
| Поиск | Dense only | Dense / BM25 | Dense + BM25 (RRF) |
| Без Docker | Да (persist_dir) | Нет | Нет |
| In-memory (тесты) | Нет | Да | Да |
| BM25 | Нет | Да | Да |
| Retry | Да (tenacity) | Да (tenacity) | Да (tenacity) |
