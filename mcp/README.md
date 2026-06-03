# Модуль `mcp/` — MCP-сервер RAG-платформы

MCP (Model Context Protocol) — протокол от Anthropic, позволяющий Claude Desktop и другим MCP-клиентам вызывать внешние инструменты. Сервер запускается как отдельный процесс и общается с клиентом через **stdin/stdout** в формате JSON-RPC.

Этот сервер даёт Claude прямой доступ к векторному хранилищу платформы: можно посмотреть коллекции и искать документы по смыслу.

---

## Структура файлов

```
mcp/
├── __init__.py      # Расширяет __path__ чтобы разрешить конфликт имён с SDK mcp
├── rag_server.py    # Сервер с инструментами list_collections и search
└── example.py       # Пример запуска клиента для проверки сервера
```

---

## Инструменты

### `list_collections`

Возвращает все коллекции векторного хранилища с числом векторов в каждой.

**Аргументы:** нет

**Пример ответа:**
```json
[
  { "name": "default", "vectors_count": 1024 },
  { "name": "docs",    "vectors_count": 256  }
]
```

---

### `search`

Семантический поиск по векторному хранилищу. Возвращает наиболее релевантные фрагменты.

**Аргументы:**

| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `query` | string | да | Поисковый запрос на естественном языке |
| `n_results` | integer | нет (5) | Сколько результатов вернуть |

**Пример ответа:**
```json
[
  { "rank": 1, "score": 0.9821, "text": "RAG combines search with LLM generation.", "meta": {} },
  { "rank": 2, "score": 0.9103, "text": "Vector databases store embeddings.",        "meta": {} }
]
```

`score` — косинусное сходство (чем выше, тем релевантнее).

---

## Запуск

### Пример (`example.py`) — рекомендуется для проверки

Поднимает MCP-клиент, подключается к серверу и вызывает оба инструмента:

```bash
PYTHONPATH=. .venv/bin/python mcp/example.py
```

Вывод:
```
Подключились к MCP-серверу

─── Доступные инструменты ───
  list_collections: Вернуть список коллекций...
  search: Семантический поиск по коллекции...

─── list_collections ───
[
  { "name": "default", "vectors_count": 0 }
]

─── search ───
Запрос: 'что такое RAG?'
[]
```

Пустой `search` — это нормально, коллекция пустая. После загрузки документов результаты появятся.

Как устроен `example.py`:
```python
# Запускает rag_server.py как subprocess и подключается к нему через stdio
async with stdio_client(SERVER) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()          # обязательное рукопожатие протокола
        tools = await session.list_tools()  # запросить список инструментов
        result = await session.call_tool("search", {"query": "...", "n_results": 3})
```

`initialize()` — обязательный первый шаг протокола MCP. Без него сервер отклонит любой другой запрос.

---

### Напрямую (для отладки процесса)

```bash
PYTHONPATH=. .venv/bin/python mcp/rag_server.py
```

Сервер ждёт JSON-RPC сообщений через stdin. Ручной ввод неудобен — сервер ожидает строго определённую последовательность сообщений. Используй `example.py` или Claude Desktop вместо этого.

### Подключение к Claude Desktop

Добавить в `~/claude_desktop_config.json` (создать если не существует):

```json
{
  "mcpServers": {
    "rag-platform": {
      "command": "/абсолютный/путь/к/.venv/bin/python",
      "args": ["/абсолютный/путь/к/rag-platform/mcp/rag_server.py"],
      "env": {
        "PYTHONPATH": "/абсолютный/путь/к/rag-platform"
      }
    }
  }
}
```

После перезапуска Claude Desktop инструменты `list_collections` и `search` станут доступны в чате.

**Пример запроса к Claude:**
> "Найди документы о векторных базах данных в моём хранилище"

Claude сам вызовет инструмент `search` с нужным запросом.

---

## Как устроен сервер (`rag_server.py`)

### Жизненный цикл

```
python mcp/rag_server.py
    │
    ├── импорт модуля — компоненты НЕ создаются (ленивая инициализация)
    │
    ├── Server("rag-platform") — регистрирует обработчики
    │
    └── main() → stdio_server()
          │
          ├── ждёт list_tools от клиента
          │       └── возвращает описание инструментов
          │
          └── ждёт call_tool от клиента
                ├── первый вызов → _get_components() создаёт VectorDB + embed_fn
                └── выполняет инструмент → возвращает JSON
```

### Ленивая инициализация

Компоненты (VectorDB, embed_fn) создаются при **первом вызове инструмента**, а не при запуске:

```python
_vector_db = None
_embed_fn  = None

def _get_components():
    global _vector_db, _embed_fn
    if _vector_db is None:
        _vector_db = build_vector_db(settings)  # подключается к Chroma/Qdrant
    if _embed_fn is None:
        _embed_fn = build_embed_fn(settings)
    return _vector_db, _embed_fn
```

Это нужно по двум причинам:
1. Импорт файла в тестах не поднимает соединения с БД
2. MCP-клиент импортирует файл при старте — без ленивой инициализации падал бы при отсутствии Chroma

### Обработка инструментов

```python
@server.list_tools()
async def handle_list_tools():
    # MCP-клиент вызывает это при подключении, чтобы узнать доступные инструменты.
    return [Tool(name="list_collections", ...), Tool(name="search", ...)]

@server.call_tool()
async def handle_call_tool(name, arguments):
    # Вызывается когда клиент хочет выполнить инструмент.
    if name == "list_collections": ...
    if name == "search": ...
```

Ответ всегда возвращается как `list[TextContent]` — список текстовых блоков. Мы кладём туда JSON-строку.

### Конфликт имён с SDK

Наша папка `mcp/` и pip-пакет `mcp` имеют одинаковое имя. Python нашёл бы нашу папку раньше и `import mcp.types` упал бы с `ModuleNotFoundError`.

Решение в `mcp/__init__.py`:

```python
# Расширяем __path__ чтобы Python находил mcp.types, mcp.server и т.д.
# из установленного SDK, а не только из нашей папки.
_site_mcp = [str(p) for p in Path(sys.prefix).glob("lib/*/site-packages/mcp") ...]
__path__ = list(__path__) + _site_mcp
```

Файл назван `rag_server.py` (а не `server.py`), чтобы не затенять `mcp.server` из SDK.

---

## Как добавить новый инструмент

1. Добавить описание в `handle_list_tools()`:

```python
types.Tool(
    name="my_tool",
    description="Что делает инструмент",
    inputSchema={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Описание параметра"}
        },
        "required": ["param"],
    },
)
```

2. Добавить ветку в `handle_call_tool()`:

```python
if name == "my_tool":
    return await _tool_my_tool(arguments["param"])
```

3. Реализовать функцию:

```python
async def _tool_my_tool(param: str) -> list[types.TextContent]:
    result = {"param": param, "answer": "..."}
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
```

---

## Переменные окружения

Сервер читает те же настройки что и API — через `config.py`:

| Переменная | Описание |
|---|---|
| `VECTOR_STORE_BACKEND` | `chroma` или `qdrant` |
| `CHROMA_PERSIST_DIR` | Путь к локальным файлам ChromaDB (без сервера) |
| `QDRANT_URL` | URL Qdrant-сервера |
| `OPENAI_API_KEY` | Если задан — используются реальные OpenAI эмбеддинги для `search` |
