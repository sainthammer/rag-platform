"""MCP-сервер RAG-платформы — stdio transport.

Инструменты:
    list_collections  — перечислить коллекции векторного хранилища.
    search            — семантический поиск по коллекции.

Запуск:
    PYTHONPATH=. .venv/bin/python mcp/rag_server.py

Подключение в Claude Desktop (~/claude_desktop_config.json):
    {
      "mcpServers": {
        "rag-platform": {
          "command": "/path/to/.venv/bin/python",
          "args": ["/path/to/rag-platform/mcp/rag_server.py"]
        }
      }
    }
"""

import asyncio
import json
from typing import TYPE_CHECKING

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from config import build_embedding_service, build_vector_db, settings

if TYPE_CHECKING:
    from vector_store.ports import VectorDB

# ---------------------------------------------------------------------------
# Ленивая инициализация компонентов
# ---------------------------------------------------------------------------
# Подключение к векторному хранилищу устанавливается при первом вызове
# инструмента, а не при импорте модуля. Это позволяет:
#   1. Импортировать файл в тестах без запущенного Chroma/Qdrant.
#   2. Использовать сервер как библиотеку без побочных эффектов при импорте.

_vector_db: "VectorDB | None" = None
_embed_fn = None


def _get_components() -> "tuple[VectorDB, object]":
    """Инициализировать и закешировать компоненты при первом вызове."""
    global _vector_db, _embed_fn
    if _vector_db is None:
        _vector_db = build_vector_db(settings)
    if _embed_fn is None:
        _embed_fn = build_embedding_service(settings).embed
    return _vector_db, _embed_fn


# ---------------------------------------------------------------------------
# Сервер
# ---------------------------------------------------------------------------

server = Server("rag-platform")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Описание доступных инструментов для MCP-клиента."""
    return [
        types.Tool(
            name="list_collections",
            description="Вернуть список коллекций векторного хранилища с числом векторов.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="search",
            description=(
                "Семантический поиск по коллекции. "
                "Возвращает n_results наиболее релевантных фрагментов."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на естественном языке.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Число возвращаемых результатов (по умолчанию 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Обработчик вызова инструмента от MCP-клиента.

    Args:
        name:      Имя инструмента (``list_collections`` или ``search``).
        arguments: Аргументы, переданные клиентом.

    Returns:
        Список ``TextContent`` с результатом в виде JSON-строки.
    """
    if name == "list_collections":
        return await _tool_list_collections()

    if name == "search":
        query = arguments.get("query", "")
        n_results = int(arguments.get("n_results", 5))
        return await _tool_search(query, n_results)

    return [types.TextContent(type="text", text=f"Неизвестный инструмент: {name!r}")]


# ---------------------------------------------------------------------------
# Реализация инструментов
# ---------------------------------------------------------------------------

async def _tool_list_collections() -> list[types.TextContent]:
    """Список коллекций из текущего бэкенда."""
    vector_db, _ = _get_components()
    backend = settings.vector_store_backend
    client = vector_db.client  # type: ignore[attr-defined]

    if backend == "chroma":
        cols = client.list_collections()
        result = [{"name": c.name, "vectors_count": c.count()} for c in cols]
    else:
        cols = client.get_collections().collections
        result = []
        for c in cols:
            try:
                count = client.count(c.name).count
            except Exception:
                count = 0
            result.append({"name": c.name, "vectors_count": count})

    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _tool_search(query: str, n_results: int) -> list[types.TextContent]:
    """Семантический поиск: векторизуем запрос → ищем в хранилище."""
    vector_db, embed_fn = _get_components()
    embedding = embed_fn(query)  # type: ignore[operator]
    results = vector_db.search(embedding, n_results=n_results)

    output = []
    for i, (doc, dist, meta) in enumerate(
        zip(results.documents, results.distances, results.metadatas), start=1
    ):
        output.append({"rank": i, "score": round(dist, 4), "text": doc, "meta": meta})

    return [types.TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
