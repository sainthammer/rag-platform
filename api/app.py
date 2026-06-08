"""FastAPI-приложение: точка сборки всех роутеров и lifespan."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from api.middleware.auth import require_auth
from api.routers import collections, health
from config import build_embedding_service, build_llm_provider, build_vector_db, settings
from retrieval.pipeline import RAGPipeline

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("━━━ RAG Platform starting ━━━")
    logger.info("LLM provider : %s", settings.llm_provider)
    logger.info("Vector store : %s", settings.vector_store_backend)
    logger.info("API key      : %s", settings.api_secret_key)
    logger.info("JWT algorithm: %s", settings.jwt_algorithm)

    llm = build_llm_provider(settings)
    vector_db = build_vector_db(settings)
    store_client = vector_db.client  # type: ignore[attr-defined]
    embed_service = build_embedding_service(settings)
    pipeline = RAGPipeline(
        llm=llm,
        vector_db_factory=lambda name: build_vector_db(settings, collection=name),
        embed_fn=embed_service.embed,
    )

    app.state.llm = llm
    app.state.vector_db = vector_db
    app.state.store_client = store_client
    app.state.store_backend = settings.vector_store_backend
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
        {"name": "collections", "description": "Управление коллекциями векторного хранилища"},
    ],
)

app.include_router(health.router, prefix="/v1")

app.include_router(
    collections.router,
    prefix="/v1",
    dependencies=[Depends(require_auth)],
)
