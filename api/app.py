"""FastAPI-приложение: точка сборки всех роутеров и lifespan."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.middleware.auth import require_auth
from api.middleware.request_id import RequestIDMiddleware
from api.routers import ask, collections, eval, health, ingest, metrics, search
from config import build_embedding_service, build_llm_provider, build_vector_db, settings
from embeddings.ports import EmbeddingService
from observability.tracing import instrument_fastapi
from llm.prompt_templates import BASE
from retrieval.pipeline import RAGPipeline

logger = logging.getLogger("uvicorn.error")


_SEED_FILE_BATCH = 20  # файлов чанкается параллельно за один раз


async def _seed_docs_task(embed_service: EmbeddingService, collection: str, docs_dir: str = "docs") -> None:
    """Загрузить все .md из docs/ в коллекцию, если она пустая.

    Алгоритм:
      1. Чанкаем _SEED_FILE_BATCH файлов параллельно (asyncio.gather + to_thread).
      2. Собираем все тексты батча в один список → один вызов embed_batch.
         Для sentence-transformers одна большая матрица быстрее N маленьких.
      3. Один вызов db.add на весь батч.
    """
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

    total_batches = (len(md_files) + _SEED_FILE_BATCH - 1) // _SEED_FILE_BATCH
    logger.info(
        "Seed: %d файлов, батчи по %d → %d итераций",
        len(md_files), _SEED_FILE_BATCH, total_batches,
    )

    total_chunks = 0
    for batch_idx in range(0, len(md_files), _SEED_FILE_BATCH):
        batch_paths = md_files[batch_idx : batch_idx + _SEED_FILE_BATCH]

        # Шаг 1 — параллельный чанкинг N файлов
        results = await asyncio.gather(
            *[asyncio.to_thread(chunking_ingest, p, strategy="by_header") for p in batch_paths],
            return_exceptions=True,
        )

        # Шаг 2 — собираем все чанки батча в плоские списки
        all_texts: list[str] = []
        all_ids: list[str] = []
        all_metas: list[dict] = []
        for path, result in zip(batch_paths, results):
            if isinstance(result, Exception):
                logger.warning("Seed: ошибка при чанкинге %s: %s", path.name, result)
                continue
            if not result:
                continue
            rel = path.relative_to(docs_path).as_posix()
            for idx, chunk in enumerate(result):
                all_texts.append(chunk.text)
                all_ids.append(f"{rel}_chunk{idx}")
                all_metas.append(chunk.metadata)

        if not all_texts:
            continue

        # Шаг 3 — один большой embed_batch на весь батч файлов
        try:
            embeddings = await asyncio.to_thread(embed_service.embed_batch, all_texts)
        except Exception as exc:
            logger.warning("Seed: ошибка embed_batch в батче %d: %s", batch_idx // _SEED_FILE_BATCH + 1, exc)
            continue

        # Шаг 4 — один db.add на весь батч
        await asyncio.to_thread(db.add, all_ids, embeddings, all_texts, all_metas)
        total_chunks += len(all_texts)
        logger.info(
            "Seed: батч %d/%d — %d чанков (итого %d)",
            batch_idx // _SEED_FILE_BATCH + 1, total_batches, len(all_texts), total_chunks,
        )

    logger.info("Seed: готово — %d чанков из %d файлов", total_chunks, len(md_files))


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
        template=BASE,
        score_threshold=0.75,
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
    title="RAG Platform",
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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(RequestIDMiddleware)

instrument_fastapi(app)
