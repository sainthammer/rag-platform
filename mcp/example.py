"""Пример взаимодействия с MCP-сервером через встроенный клиент.

Запуск (без реального ChromaDB/Qdrant):
    PYTHONPATH=. CHROMA_PERSIST_DIR=.tmp/demo-chroma .venv/bin/python mcp/example.py

Требования:
    - ChromaDB с данными (CHROMA_PERSIST_DIR) или запущенный сервер (CHROMA_HOST/CHROMA_PORT).
    - Для search: настроенный EMBEDDING_PROVIDER с моделью, совместимой с данными в коллекции.
"""

import asyncio
import json
import os

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


SERVER = StdioServerParameters(
    command=".venv/bin/python",
    args=["mcp/rag_server.py"],
    env={**os.environ, "PYTHONPATH": "."},
)


def _parse_result(result) -> object:
    """Распарсить ответ инструмента; вернуть сырой текст при ошибке."""
    if not result.content:
        return "<пустой ответ>"
    text = result.content[0].text
    if not text:
        return "<пустой текст>"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def main() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Подключились к MCP-серверу\n")

            # ---- list_tools ----
            print("─── Доступные инструменты ───")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  {tool.name}: {tool.description}")
            print()

            # ---- list_collections ----
            print("─── list_collections ───")
            result = await session.call_tool("list_collections", {})
            data = _parse_result(result)
            print(json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (list, dict)) else data)
            print()

            # ---- search ----
            print("─── search ───")
            query = "что такое RAG?"
            print(f"Запрос: {query!r}")
            result = await session.call_tool("search", {"query": query, "n_results": 3})
            data = _parse_result(result)
            print(json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (list, dict)) else data)


if __name__ == "__main__":
    asyncio.run(main())
