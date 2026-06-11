"""FastAPI-приложение: точка сборки всех роутеров и lifespan."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from api.middleware.auth import require_auth
from api.routers import ask, collections, eval, health, ingest, metrics, search
from config import build_embedding_service, build_llm_provider, build_vector_db, settings
from embeddings.ports import EmbeddingService
from observability.tracing import instrument_fastapi
from retrieval.pipeline import RAGPipeline

logger = logging.getLogger("uvicorn.error")


async def _seed_docs_task(embed_service: EmbeddingService, collection: str, docs_dir: str = "docs") -> None:
    """Загрузить все .md из docs/ в коллекцию, если она пустая."""
    from chunking import ingest as chunking_ingest

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return

    db = build_vector_db(settings, collection=collection)
    if await asyncio.to_thread(db.count) > 0:
        logger.info("Seed: коллекция %r уже заполнена, пропускаем", collection)
        return

    md_files = sorted(docs_path.rglob("*.md"))
    if not md_files:
        return

    logger.info("Seed: индексируем %d .md файлов → коллекция %r", len(md_files), collection)
    total = 0
    for path in md_files:
        try:
            chunks = await asyncio.to_thread(chunking_ingest, path, strategy="by_header")
            if not chunks:
                continue
            texts = [c.text for c in chunks]
            embeddings = await asyncio.to_thread(embed_service.embed_batch, texts)
            rel = path.relative_to(docs_path).as_posix()
            ids = [f"{rel}_chunk{i}" for i in range(len(chunks))]
            metadatas = [c.metadata for c in chunks]
            await asyncio.to_thread(db.add, ids, embeddings, texts, metadatas)
            total += len(chunks)
            logger.info("Seed: %s → %d чанков", path.name, len(chunks))
        except Exception as exc:
            logger.warning("Seed: не удалось индексировать %s: %s", path.name, exc)

    logger.info("Seed: готово — %d чанков из %d файлов", total, len(md_files))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("━━━ RAG Platform starting ━━━")
    logger.info("LLM provider : %s", settings.llm_provider)
    logger.info("Vector store : %s", settings.vector_store_backend)

    llm = build_llm_provider(settings)
    vector_db = build_vector_db(settings)
    embed_service = build_embedding_service(settings)
    pipeline = RAGPipeline(
        llm=llm,
        vector_db_factory=lambda name: build_vector_db(settings, collection=name),
        embed_fn=embed_service.embed,
    )

    app.state.llm = llm
    app.state.vector_db = vector_db
    app.state.store_client = vector_db.client  # type: ignore[attr-defined]
    app.state.store_backend = settings.vector_store_backend
    app.state.embed_service = embed_service
    app.state.pipeline = pipeline

    asyncio.create_task(_seed_docs_task(embed_service, settings.default_collection))

    yield


app = FastAPI(
    title="RAG Platform API",
    version="0.1.0",
    description=(
        "RAG-платформа с поддержкой нескольких LLM-провайдеров и векторных хранилищ.\n\n"
        "**Аутентификация** — нажми **Authorize** и введи `API_SECRET_KEY` из `.env`:\n"
        "- поле `X-API-Key` — прямой ключ\n"
        "- поле `Bearer` — JWT-токен"
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Проверка состояния системы (без аутентификации)"},
        {"name": "ingest", "description": "Загрузка документов в векторное хранилище"},
        {"name": "search", "description": "Семантический поиск по коллекции"},
        {"name": "ask", "description": "Полный RAG-цикл: retrieval + generation"},
        {"name": "eval", "description": "Оценка качества пайплайна через RAGAS"},
        {"name": "collections", "description": "Управление коллекциями"},
        {"name": "infra", "description": "Health check и метрики Prometheus"},
    ],
)

# Без аутентификации
app.include_router(health.router, prefix="/v1")
app.include_router(metrics.router, prefix="/v1")

# С аутентификацией
_auth = [Depends(require_auth)]
app.include_router(ingest.router, prefix="/v1", dependencies=_auth)
app.include_router(search.router, prefix="/v1", dependencies=_auth)
app.include_router(ask.router, prefix="/v1", dependencies=_auth)
app.include_router(eval.router, prefix="/v1", dependencies=_auth)
app.include_router(collections.router, prefix="/v1", dependencies=_auth)

instrument_fastapi(app)
