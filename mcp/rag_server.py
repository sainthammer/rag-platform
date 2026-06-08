"""MCP-сервер RAG-платформы.

Инструменты:
    list_collections   — перечислить коллекции векторного хранилища.
    search             — семантический поиск по коллекции.
    ingest_document    — загрузить текст и проиндексировать в коллекцию.
    ask                — задать вопрос RAG-пайплайну.

Транспорты::

    # stdio (default, для Claude Desktop)
    PYTHONPATH=. .venv/bin/python mcp/rag_server.py

    # HTTP (Streamable HTTP, для MCP-клиентов с поддержкой HTTP)
    PYTHONPATH=. .venv/bin/python mcp/rag_server.py --transport http --port 8001

Подключение в Claude Desktop (~/claude_desktop_config.json)::

    {
      "mcpServers": {
        "rag-platform": {
          "command": "/path/to/.venv/bin/python",
          "args": ["/path/to/rag-platform/mcp/rag_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from chunking import chunk_text
from config import build_embedding_service, build_llm_provider, build_vector_db, settings

if TYPE_CHECKING:
    from vector_store.ports import VectorDB

# ---------------------------------------------------------------------------
# Ленивая инициализация компонентов
# ---------------------------------------------------------------------------

_vector_db: "VectorDB | None" = None
_embed_fn = None
_pipeline = None


def _get_components() -> "tuple[VectorDB, object]":
    global _vector_db, _embed_fn
    if _vector_db is None:
        _vector_db = build_vector_db(settings)
    if _embed_fn is None:
        _embed_fn = build_embedding_service(settings).embed
    return _vector_db, _embed_fn


def _get_pipeline():
    global _pipeline, _embed_fn
    if _pipeline is None:
        from retrieval.pipeline import RAGPipeline
        if _embed_fn is None:
            _embed_fn = build_embedding_service(settings).embed
        _pipeline = RAGPipeline(
            llm=build_llm_provider(settings),
            vector_db_factory=lambda name: build_vector_db(settings, collection=name),
            embed_fn=_embed_fn,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Сервер
# ---------------------------------------------------------------------------

server = Server("rag-platform")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_collections",
            description="Вернуть список коллекций векторного хранилища с числом векторов.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="search",
            description="Семантический поиск по коллекции. Возвращает n_results релевантных фрагментов.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос."},
                    "collection": {"type": "string", "description": "Имя коллекции.", "default": "default"},
                    "n_results": {"type": "integer", "description": "Число результатов.", "default": 5},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ingest_document",
            description="Проиндексировать текст документа в векторное хранилище.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Текст документа для индексации."},
                    "collection": {"type": "string", "description": "Имя коллекции.", "default": "default"},
                    "doc_id": {"type": "string", "description": "Идентификатор документа.", "default": "doc"},
                    "chunk_size": {"type": "integer", "description": "Размер чанка.", "default": 500},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="ask",
            description="Задать вопрос RAG-пайплайну. Возвращает ответ с источниками.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Вопрос на естественном языке."},
                    "collection": {"type": "string", "description": "Имя коллекции.", "default": "default"},
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "list_collections":
        return await _tool_list_collections()
    if name == "search":
        return await _tool_search(
            query=arguments["query"],
            collection=arguments.get("collection", settings.default_collection),
            n_results=int(arguments.get("n_results", 5)),
        )
    if name == "ingest_document":
        return await _tool_ingest_document(
            text=arguments["text"],
            collection=arguments.get("collection", settings.default_collection),
            doc_id=arguments.get("doc_id", "doc"),
            chunk_size=int(arguments.get("chunk_size", 500)),
        )
    if name == "ask":
        return await _tool_ask(
            question=arguments["question"],
            collection=arguments.get("collection", settings.default_collection),
        )
    return [types.TextContent(type="text", text=f"Неизвестный инструмент: {name!r}")]


# ---------------------------------------------------------------------------
# Реализация инструментов
# ---------------------------------------------------------------------------

async def _tool_list_collections() -> list[types.TextContent]:
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


async def _tool_search(query: str, collection: str, n_results: int) -> list[types.TextContent]:
    _, embed_fn = _get_components()
    db = build_vector_db(settings, collection=collection)
    embedding = embed_fn(query)  # type: ignore[operator]
    results = db.search(embedding, n_results=n_results)

    output = []
    for i, (doc_id, doc, dist, meta) in enumerate(
        zip(results.ids, results.documents, results.distances, results.metadatas), start=1
    ):
        score = round(1.0 / (1.0 + dist), 4)
        output.append({"rank": i, "id": doc_id, "score": score, "text": doc, "meta": meta})

    return [types.TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]


async def _tool_ingest_document(
    text: str, collection: str, doc_id: str, chunk_size: int
) -> list[types.TextContent]:
    _, embed_fn = _get_components()
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=50)
    if not chunks:
        return [types.TextContent(type="text", text=json.dumps({"indexed": 0, "collection": collection}))]

    embeddings = [embed_fn(chunk) for chunk in chunks]  # type: ignore[operator]
    ids = [f"{doc_id}_chunk{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]

    db = build_vector_db(settings, collection=collection)
    db.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    result = {"indexed": len(chunks), "collection": collection, "doc_id": doc_id}
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _tool_ask(question: str, collection: str) -> list[types.TextContent]:
    pipeline = _get_pipeline()
    response = await pipeline.ask(question, collection)
    result = {
        "answer": response.answer,
        "confidence": round(response.confidence, 4),
        "latency_ms": round(response.latency_ms, 1),
        "sources": [
            {"text": s.text[:200], "score": round(s.score, 4), "metadata": s.metadata}
            for s in response.sources
        ],
    }
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Platform MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Транспорт: stdio (default, для Claude Desktop) или http",
    )
    parser.add_argument("--port", type=int, default=8001, help="Порт для HTTP-транспорта")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для HTTP-транспорта")
    return parser.parse_args()


async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def _run_http(host: str, port: int) -> None:
    import contextlib
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.routing import Mount, Route

    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        stateless=True,
    )

    async def handle_mcp(scope, receive, send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            yield

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=handle_mcp)],
        lifespan=lifespan,
    )

    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main() -> None:
    args = _parse_args()
    if args.transport == "http":
        print(f"MCP HTTP server starting on http://{args.host}:{args.port}/mcp")
        asyncio.run(_run_http(args.host, args.port))
    else:
        asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
