"""POST /v1/ingest — загрузка документа (файл или URL) в векторное хранилище.
GET  /v1/ingest/{job_id} — статус задачи индексации.
"""

from __future__ import annotations

import asyncio
import io
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from api.deps import get_embed_service
from api.jobs import Job, get_job, new_job, update_job
from api.schemas import IngestJobResponse, IngestStatusResponse
from chunking import chunk_text
from config import build_vector_db, settings
from embeddings.ports import EmbeddingService

router = APIRouter(tags=["ingest"])


# ---------------------------------------------------------------------------
# Фоновая задача
# ---------------------------------------------------------------------------

async def _ingest_task(
    job: Job,
    text: str,
    collection: str,
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_service: EmbeddingService,
) -> None:
    await update_job(job.job_id, "running")
    try:
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            await update_job(job.job_id, "done", {"chunks_indexed": 0, "collection": collection})
            return

        embeddings = embed_service.embed_batch(chunks)
        ids = [f"{doc_id}_chunk{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]

        db = build_vector_db(settings, collection=collection)
        db.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

        await update_job(job.job_id, "done", {"chunks_indexed": len(chunks), "collection": collection})
    except Exception as exc:  # noqa: BLE001
        await update_job(job.job_id, "error", error=str(exc))


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Загрузить документ в векторное хранилище",
)
async def ingest(
    request: Request,
    embed_service: Annotated[EmbeddingService, Depends(get_embed_service)],
    file: Annotated[UploadFile | None, File(description="Текстовый файл для индексации")] = None,
    url: Annotated[str | None, Form(description="URL документа для загрузки")] = None,
    text: Annotated[str | None, Form(description="Текст для индексации напрямую")] = None,
    collection: Annotated[str, Form()] = "default",
    doc_id: Annotated[str | None, Form()] = None,
    chunk_size: Annotated[int, Form(ge=50, le=4000)] = 500,
    chunk_overlap: Annotated[int, Form(ge=0, le=500)] = 50,
) -> IngestJobResponse:
    """Принять документ и запустить индексацию в фоне.

    Источник — ровно один из: `file`, `url`, `text`.
    Возвращает `job_id` для отслеживания прогресса через GET /v1/ingest/{job_id}.
    """
    sources = [x for x in (file, url, text) if x is not None]
    if len(sources) != 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Укажите ровно один источник: file, url или text.",
        )

    # Читаем содержимое
    if file is not None:
        raw = await file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Файл должен быть в кодировке UTF-8.",
            )
        effective_doc_id = doc_id or (file.filename or "upload")
    elif url is not None:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            content = resp.text
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Не удалось загрузить URL: {exc}",
            )
        effective_doc_id = doc_id or url
    else:
        content = text  # type: ignore[assignment]
        effective_doc_id = doc_id or "text_input"

    job = new_job()
    asyncio.create_task(
        _ingest_task(
            job=job,
            text=content,
            collection=collection,
            doc_id=effective_doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_service=embed_service,
        )
    )
    return IngestJobResponse(job_id=job.job_id, status="pending")


@router.get(
    "/ingest/{job_id}",
    response_model=IngestStatusResponse,
    summary="Статус задачи индексации",
    responses={404: {"description": "Задача не найдена"}},
)
async def ingest_status(job_id: str) -> IngestStatusResponse:
    """Вернуть текущий статус задачи индексации по её job_id."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id!r} не найден")
    return IngestStatusResponse(
        job_id=job.job_id,
        status=job.status,
        chunks_indexed=job.result.get("chunks_indexed", 0),
        collection=job.result.get("collection", ""),
        error=job.error,
    )
