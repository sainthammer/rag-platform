# Модуль 10 — mcp/

**Файл:** `mcp/rag_server.py`

MCP-сервер — интеграция RAG-платформы с Claude Desktop и другими AI-агентами как набор инструментов.

## Что такое MCP

**MCP (Model Context Protocol)** — открытый стандарт Anthropic для подключения внешних инструментов к AI-агентам. Claude Desktop, Claude Code и другие MCP-клиенты умеют вызывать инструменты MCP-серверов.

Аналогия: как REST API позволяет браузеру общаться с сервером, MCP позволяет LLM вызывать внешние функции.

## Четыре инструмента

| Инструмент | Описание |
|---|---|
| `list_collections` | Показать все коллекции в векторной БД с числом векторов |
| `search` | Семантический поиск по коллекции |
| `ingest_document` | Загрузить текст и проиндексировать в коллекцию |
| `ask` | Задать вопрос RAG-пайплайну, получить ответ с источниками |

Когда Claude Desktop подключён к этому MCP-серверу, можно в обычном чате написать: «Найди информацию об авторизации в нашей документации» — и Claude сам вызовет `search`, получит результаты и ответит.

## Два транспорта

### stdio (для Claude Desktop)

Клиент запускает сервер как subprocess и общается через stdin/stdout:

```bash
PYTHONPATH=. .venv/bin/python mcp/rag_server.py
```

Конфиг в `~/claude_desktop_config.json`:
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

### HTTP (для сетевых MCP-клиентов)

```bash
PYTHONPATH=. .venv/bin/python mcp/rag_server.py --transport http --port 8001
```

Запускает Starlette-приложение, MCP-сообщения доступны на `http://localhost:8001/mcp`.

## Ленивая инициализация

Компоненты создаются только при первом вызове инструмента:

```python
_vector_db: VectorDB | None = None
_embed_fn = None
_pipeline = None

def _get_components():
    global _vector_db, _embed_fn
    if _vector_db is None:
        _vector_db = build_vector_db(settings)    # ← только при первом вызове
    if _embed_fn is None:
        _embed_fn = build_embedding_service(settings).embed
    return _vector_db, _embed_fn
```

Это важно для stdio-транспорта: процесс стартует мгновенно, тяжёлые компоненты (загрузка embedding-модели) создаются только когда Claude реально вызывает инструмент.

## Реализация инструментов

### `list_collections`

```python
# Для Chroma:
cols = client.list_collections()
result = [{"name": c.name, "vectors_count": c.count()} for c in cols]

# Для Qdrant:
cols = client.get_collections().collections
result = [{"name": c.name, "vectors_count": count} for c in cols]
```

### `search`

```python
embedding = embed_fn(query)
results = db.search(embedding, n_results=n_results)
# Возвращает JSON: rank, id, score, text, meta
```

### `ingest_document`

```python
chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=50)
embeddings = [embed_fn(chunk) for chunk in chunks]
db.add(ids=[...], embeddings=embeddings, documents=chunks, metadatas=[...])
# Возвращает: {"indexed": 5, "collection": "docs", "doc_id": "manual"}
```

### `ask`

```python
pipeline = _get_pipeline()
response = await pipeline.ask(question, collection)
result = {
    "answer": response.answer,
    "confidence": round(response.confidence, 4),
    "latency_ms": round(response.latency_ms, 1),
    "sources": [{"text": s.text[:200], "score": s.score} for s in response.sources],
}
```

## Все ответы — JSON в TextContent

MCP требует возвращать `list[TextContent]`. Каждый инструмент возвращает один элемент:

```python
return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
```

`ensure_ascii=False` — чтобы кириллица не превращалась в `код`.

## Пример сессии с Claude Desktop

```
Пользователь: «Что у нас написано про настройку Redis?»

Claude: (вызывает search с query="настройка Redis", collection="docs")
→ [{"rank": 1, "text": "Redis настраивается через переменную REDIS_URL...", "score": 0.87}]

Claude: «В вашей документации написано, что Redis настраивается через переменную 
окружения REDIS_URL. По умолчанию используется localhost:6379...»
```
