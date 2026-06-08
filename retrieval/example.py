"""Smoke-проверка RAGPipeline без внешних сервисов.

Запуск:
    PYTHONPATH=. .venv/bin/python -m retrieval.example
"""

import asyncio
import hashlib
import math
from typing import AsyncGenerator

from llm.llm_dataclasses import Message
from llm.ports import LLMProvider
from retrieval.pipeline import RAGPipeline
from vector_store.adapters import ChromaDB

DOCS = [
    "RAG (Retrieval-Augmented Generation) объединяет поиск по документам и генерацию ответа LLM.",
    "ChromaDB — векторная база данных, которая хранит embedding-векторы и ищет по ним.",
    "Embedding-вектор — числовое представление текста, где похожие тексты близки в пространстве.",
    "Fallback срабатывает, когда retrieval не нашёл релевантных чанков выше порога.",
]


# --- Fake embedding (детерминированный, без сети) ---

def fake_embed(text: str, dim: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm > 0 else values


# --- Fake LLM (имитирует ответ модели) ---

class FakeLLM(LLMProvider):
    async def complete(
        self, messages: list[Message], stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        context_line = next(
            (m.content for m in messages if m.role == "system"), ""
        ).split("\n")[3]  # первая строка контекста
        answer = f"[FakeLLM] Контекст получен. Первый чанк начинается: «{context_line[:60]}…»"
        if stream:
            async def _gen() -> AsyncGenerator[str, None]:
                for word in answer.split():
                    yield word + " "
                    await asyncio.sleep(0.02)
            return _gen()
        return answer


async def main() -> None:
    # Создаём in-memory Chroma и заполняем документами
    db = ChromaDB(collection="demo", persist_directory=".tmp/demo-chroma")
    db.add(
        ids=[f"doc{i}" for i in range(len(DOCS))],
        embeddings=[fake_embed(d) for d in DOCS],
        documents=DOCS,
        metadatas=[{"source": f"doc{i}.txt"} for i in range(len(DOCS))],
    )

    pipeline = RAGPipeline(
        llm=FakeLLM(),
        vector_db_factory=lambda _: db,
        embed_fn=fake_embed,
        n_results=3,
        score_threshold=0.1,
    )

    # --- ask() ---
    print("=" * 60)
    print("ask(): обычный запрос")
    print("=" * 60)
    response = await pipeline.ask("Что такое RAG?", collection="demo")
    print(f"Ответ:      {response.answer}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Latency:    {response.latency_ms:.1f} ms")
    print(f"Sources ({len(response.sources)}):")
    for s in response.sources:
        print(f"  [{s.score:.3f}] {s.metadata['source']}  {s.text[:55]}…")

    # --- ask() fallback ---
    print()
    print("=" * 60)
    print("ask() fallback: score_threshold=1.1 (невозможный порог)")
    print("=" * 60)
    pipeline_strict = RAGPipeline(
        llm=FakeLLM(),
        vector_db_factory=lambda _: db,
        embed_fn=fake_embed,
        score_threshold=1.1,
    )
    fb = await pipeline_strict.ask("вопрос без шансов", collection="demo")
    print(f"Ответ:   {fb.answer}")
    print(f"Sources: {fb.sources}")

    # --- ask_stream() ---
    print()
    print("=" * 60)
    print("ask_stream(): токены в реальном времени")
    print("=" * 60)
    print("Токены: ", end="", flush=True)
    async for token in pipeline.ask_stream("Что такое embedding?", "demo"):
        print(token, end="", flush=True)
    print()

    # --- ask_multi() ---
    print()
    print("=" * 60)
    print("ask_multi(): три коллекции параллельно")
    print("=" * 60)
    responses = await pipeline.ask_multi("ChromaDB", ["col-a", "col-b", "col-c"])
    for col, r in zip(["col-a", "col-b", "col-c"], responses):
        print(f"  {col}: confidence={r.confidence:.3f}, sources={len(r.sources)}")


if __name__ == "__main__":
    asyncio.run(main())
