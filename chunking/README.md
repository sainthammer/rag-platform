# Модуль `chunking/`

`chunking/` отвечает за первый этап RAG ingestion pipeline: загрузить документ,
привести текст к единому виду и разбить его на чанки, которые дальше можно
передавать в `embeddings/` и сохранять в `vector_store/`.

## Роль в RAG pipeline

Поток индексации документа выглядит так:

```text
файл (.txt/.md/.html/.pdf)
    -> chunking.ingest()
    -> list[Chunk]
    -> embeddings.embed_batch([chunk.text, ...])
    -> vector_store.add(ids, embeddings, documents, metadatas)
```

`chunking` не строит embedding-векторы и не пишет данные в vector store. Его
контракт заканчивается на списке `Chunk`, где каждый элемент содержит:

- `text` - текстовый фрагмент;
- `metadata` - метаданные источника и разбиения;
- `id` - стабильный идентификатор чанка.

## Структура файлов

```text
chunking/
├── __init__.py       # публичный API модуля
├── chunkers.py       # стратегии разбиения текста
├── dataclasses.py    # структура Chunk
├── example.py        # интерактивный пример запуска
├── loaders.py        # загрузчики TXT, Markdown, HTML, PDF
├── pipeline.py       # функция ingest()
├── ports.py          # абстрактные интерфейсы DocumentLoader и Chunker
└── utils.py          # нормализация текста, hash, sliding windows
```

## Быстрый старт

```python
from chunking import ingest

chunks = ingest(
    "docs/example.md",
    strategy="by_header",
    chunk_size=1000,
)

for chunk in chunks:
    print(chunk.id, chunk.metadata["chunk_strategy"], chunk.text[:80])
```

## Пример запуска

Базовый пример не требует внешних сервисов, API-ключей, vector store или записи
файлов. Он использует встроенный Markdown-текст и показывает, как один и тот же
текст разбивается стратегиями `fixed`, `by_header` и `semantic`:

```bash
PYTHONPATH=. python -m chunking.example
```

По умолчанию для каждой стратегии показываются первые 10 чанков. Лимит можно
изменить:

```bash
PYTHONPATH=. python -m chunking.example --limit 20
```

По умолчанию текст чанка выводится как preview на 90 символов. Чтобы показать
полный текст чанка, используйте:

```bash
PYTHONPATH=. python -m chunking.example --preview-chars 0
```

Можно передать собственный файл и стратегию:

```bash
PYTHONPATH=. python -m chunking.example --source README.md --strategy by_header --chunk-size 500 --limit 10
```

Для HTML и PDF примера нужны зависимости из `pyproject.toml`: `beautifulsoup4`
и `pdfminer.six`.

## Загрузчики документов

Загрузчики реализуют интерфейс `DocumentLoader`:

```python
load(source_path) -> tuple[str, dict[str, object]]
```

Поддерживаемые форматы:

- `.txt`, `.text`, файл без расширения - `TextLoader`;
- `.md`, `.markdown` - `MarkdownLoader`;
- `.html`, `.htm` - `HTMLLoader`;
- `.pdf` - `PDFLoader`.

`get_loader()` выбирает загрузчик по расширению. Каждый загрузчик возвращает
нормализованный текст и метаданные:

- `source`;
- `source_name`;
- `content_type`;
- `size_bytes`;
- `modified_at`;
- `document_hash`.

## Стратегии чанкинга

### `FixedSizeChunker`

Режет текст на символьные окна фиксированного размера с overlap. Подходит как
универсальный fallback для любых документов.

Metadata чанка:

- `chunk_index`;
- `chunk_start`;
- `chunk_end`;
- `chunk_strategy="fixed"`.

### `ByHeaderChunker`

Делит Markdown-like текст по заголовкам `#`, `##`, ..., `######`. Если секция
больше `chunk_size`, она дополнительно режется fixed-window логикой.

Metadata чанка:

- `section`;
- `section_index`;
- `chunk_index`;
- `chunk_strategy="by_header"`.

### `SemanticChunker`

Сначала делит текст по абзацам. Если абзац слишком большой, дополнительно делит
его по простым границам предложений (`.`, `!`, `?`). Затем собирает чанки в
пределах `chunk_size`.

Metadata чанка:

- `chunk_index`;
- `chunk_strategy="semantic"`.

## Публичный API

```python
from chunking import (
    Chunk,
    Chunker,
    DocumentLoader,
    FixedSizeChunker,
    ByHeaderChunker,
    SemanticChunker,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    PDFLoader,
    build_chunker,
    get_loader,
    ingest,
)
```

## Связка с `embeddings/` и `vector_store/`

Минимальный пример индексации:

```python
from chunking import ingest
from embeddings import build_document_embedding_service
from vector_store.adapters import QdrantDB

chunks = ingest("docs/example.md", strategy="by_header", chunk_size=1000)

embedding_service = build_document_embedding_service()
vectors = embedding_service.embed_batch([chunk.text for chunk in chunks])

db = QdrantDB(
    collection="documents",
    vector_size=embedding_service.dimension(),
)

db.add(
    ids=[chunk.id for chunk in chunks],
    embeddings=vectors,
    documents=[chunk.text for chunk in chunks],
    metadatas=[{**chunk.metadata, "chunk_id": chunk.id} for chunk in chunks],
)
```

Порядок списков важен: `ids`, `embeddings`, `documents` и `metadatas` должны
соответствовать друг другу по индексам.

## Тесты

```bash
PYTHONPATH=. python -m pytest tests/unit/test_chunking.py -q
```

Тесты проверяют загрузчики, стратегии чанкинга и `ingest()`.
