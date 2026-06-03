"""Пример взаимодействия с MCP-сервером через встроенный клиент.

Запуск:
    PYTHONPATH=. .venv/bin/python mcp/example.py
"""

import asyncio
import json

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


SERVER = StdioServerParameters(
    command=".venv/bin/python",
    args=["mcp/rag_server.py"],
    env={"PYTHONPATH": "."},
)


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
            data = json.loads(result.content[0].text)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            print()

            # ---- search ----
            print("─── search ───")
            query = "что такое RAG?"
            print(f"Запрос: {query!r}")
            result = await session.call_tool("search", {"query": query, "n_results": 3})
            data = json.loads(result.content[0].text)
            print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
