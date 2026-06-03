"""Интерактивный smoke-тест провайдеров и RAGPipeline.

Запуск:
    PYTHONPATH=. .venv/bin/python llm/example.py
    PYTHONPATH=. .venv/bin/python llm/example.py openai
    PYTHONPATH=. .venv/bin/python llm/example.py anthropic

Что демонстрируется:
    1. Прямой вызов провайдера (без RAG) — обычный и стриминг.
    2. Полный RAG-пайплайн: in-memory Qdrant + фиктивные эмбеддинги + LLM.
    3. Все четыре шаблона промптов.

Для Ollama нужен запущенный сервер: ollama serve && ollama pull llama3.2
Для OpenAI/Anthropic нужен ключ в .env.
"""

import asyncio
import os
import sys

# ---------------------------------------------------------------------------
# Выбор провайдера из аргумента командной строки или .env
# ---------------------------------------------------------------------------

PROVIDER = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_PROVIDER", "ollama")
print(f"\n{'═' * 55}")
print(f"  LLM Example  |  провайдер={PROVIDER}")
print(f"{'═' * 55}\n")


def _build_provider():
    """Создать провайдер по значению PROVIDER."""
    if PROVIDER == "openai":
        from llm.adapters import OpenAIProvider
        return OpenAIProvider(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))

    if PROVIDER == "anthropic":
        from llm.adapters import AnthropicProvider
        return AnthropicProvider(model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"))

    from llm.adapters import OllamaProvider
    return OllamaProvider(
        model=os.getenv("LLM_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )


# ---------------------------------------------------------------------------
# 1. Прямой вызов провайдера
# ---------------------------------------------------------------------------

async def demo_direct():
    """Отправить один вопрос напрямую, без RAG."""
    from llm.llm_dataclasses import Message

    provider = _build_provider()
    query = "Что такое RAG в контексте больших языковых моделей?"
    messages = [
        Message(role="system", content="Ты лаконичный ассистент. Отвечай на русском языке в 1-2 предложениях."),
        Message(role="user", content=query),
    ]

    print("─── 1. Прямой вызов (без стриминга) ───")
    print(f"В: {query}")
    result = await provider.complete(messages, stream=False)
    print(f"О: {result}\n")

    print("─── 2. Прямой вызов (стриминг) ───")
    print(f"В: {query}")
    print("О: ", end="", flush=True)
    gen = await provider.complete(messages, stream=True)
    async for chunk in gen:  # type: ignore[union-attr]
        print(chunk, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# 2. RAGPipeline — in-memory Qdrant + фиктивные эмбеддинги
# ---------------------------------------------------------------------------

# База знаний: 5 документов о компонентах RAG-системы.
# Эмбеддинги фиктивные — для демо без реальной embedding-модели.
_DOCS = [
    "RAG (Retrieval-Augmented Generation) — это подход, при котором языковая модель дополняется поиском по базе знаний перед генерацией ответа.",
    "Векторные базы данных хранят эмбеддинги и поддерживают поиск по семантической близости с помощью метрик косинусного расстояния.",
    "Чанкинг (chunking) — разбивка документов на небольшие фрагменты перед векторизацией для более точного семантического поиска.",
    "Prompt Engineering — составление системных инструкций, направляющих модель к точным и структурированным ответам.",
    "Токенный бюджет ограничивает объём контекста, который можно передать модели за один запрос, исходя из размера контекстного окна.",
]
# Простейшие ортогональные векторы размерностью 5.
_EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]


def _build_vector_db():
    """Создать in-memory Qdrant с тестовыми данными."""
    from vector_store.adapters import QdrantDB

    db = QdrantDB("demo", vector_size=5, in_memory=True)
    db.add(
        ids=[str(i) for i in range(len(_DOCS))],
        embeddings=_EMBEDDINGS,
        documents=_DOCS,
    )
    return db


def _fake_embed(text: str) -> list[float]:
    """Фиктивный эмбеддинг: всегда возвращает вектор близкий к первому документу.

    В реальном приложении здесь будет вызов OpenAI Embeddings API или
    локальной HuggingFace-модели из модуля embeddings/.
    """
    return [0.9, 0.1, 0.0, 0.0, 0.0]


async def demo_rag_pipeline():
    """Запустить полный RAG-цикл со всеми шаблонами промптов."""
    from llm.pipeline import RAGPipeline
    from llm.prompt_templates import get_template
    from llm.token_budget import TokenBudgetManager

    provider = _build_provider()
    db = _build_vector_db()

    # TokenBudgetManager с маленьким бюджетом — покажет усечение контекста.
    budget = TokenBudgetManager(max_context_tokens=300, reserved_tokens=50)

    query = "Что такое RAG и как он работает?"

    for template_name in ("base", "strict", "citation", "multilingual"):
        print(f"─── RAGPipeline | шаблон={template_name!r} ───")
        pipeline = RAGPipeline(
            llm=provider,
            vector_db=db,
            embed_fn=_fake_embed,
            template=get_template(template_name),
            n_results=3,
            budget=budget,
        )
        result = await pipeline.run(query)
        print(f"В: {query}")
        print(f"О: {result}\n")


# ---------------------------------------------------------------------------
# 3. Стриминг через RAGPipeline
# ---------------------------------------------------------------------------

async def demo_rag_stream():
    """Показать стриминг ответа из RAGPipeline."""
    from llm.pipeline import RAGPipeline

    query = "Объясни, что такое векторные базы данных и зачем они нужны."
    print(f"─── RAGPipeline | stream=True ───")
    print(f"В: {query}")
    pipeline = RAGPipeline(
        llm=_build_provider(),
        vector_db=_build_vector_db(),
        embed_fn=_fake_embed,
        n_results=2,
    )
    print("О: ", end="", flush=True)
    gen = await pipeline.run(query, stream=True)
    async for chunk in gen:  # type: ignore[union-attr]
        print(chunk, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def main():
    await demo_direct()
    await demo_rag_pipeline()
    await demo_rag_stream()
    print("✓ Готово\n")


if __name__ == "__main__":
    asyncio.run(main())
