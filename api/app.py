"""FastAPI-приложение: точка сборки всех роутеров и lifespan."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from api.middleware.auth import require_auth
from api.routers import ask, collections, eval, health, ingest, metrics, search
from config import build_embedding_service, build_llm_provider, build_vector_db, settings
from observability.tracing import instrument_fastapi
from retrieval.pipeline import RAGPipeline

logger = logging.getLogger("uvicorn.error")


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
