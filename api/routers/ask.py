"""POST /v1/ask — полный RAG-цикл.

Режимы ответа:
  stream=false (default) → JSON AskResponse
  stream=true            → SSE-поток (text/event-stream)
    каждое событие: data: {"token": "..."}\n\n
    финальное:     data: [DONE]\n\n
"""

from __future__ import annotations

import copy
import json
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.deps import get_pipeline
from api.limiter import limiter
from api.schemas import AskRequest, AskResponse, AskSourceItem
from retrieval.pipeline import RAGPipeline


def _inline_refs(schema: dict) -> dict:
    """Раскрыть $defs/$ref в Pydantic-схеме, чтобы Swagger UI мог её разобрать."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(obj: object) -> object:
        if isinstance(obj, dict):
            if "$ref" in obj and len(obj) == 1:
                name = obj["$ref"].rsplit("/", 1)[-1]
                if name in defs:
                    return resolve(defs[name])
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(schema)  # type: ignore[return-value]


_ASK_RESPONSE_SCHEMA = _inline_refs(AskResponse.model_json_schema())

router = APIRouter(tags=["ask"])


async def _sse_generator(
    pipeline: RAGPipeline,
    question: str,
    collection: str,
) -> AsyncGenerator[str, None]:
    async for token in pipeline.ask_stream(question, collection):
        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post(
    "/ask",
    response_model=None,
    summary="Задать вопрос RAG-пайплайну",
    description=(
        "Полный RAG-цикл: embed → retrieve → generate. "
        "Лимит: **100 запросов / минуту** с одного IP. "
        "При `stream=true` возвращает `text/event-stream`."
    ),
    responses={
        200: {
            "description": "JSON AskResponse (stream=false) или SSE-поток (stream=true)",
            "content": {
                "application/json": {"schema": _ASK_RESPONSE_SCHEMA},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        },
        429: {"description": "Превышен лимит запросов (100/мин)"},
        503: {"description": "Компонент недоступен"},
    },
)
@limiter.limit("100/minute")
async def ask(
    request: Request,
    body: AskRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
):
    """Полный RAG-цикл: embed → retrieve → generate.

    При `stream=true` возвращает `text/event-stream` с токенами по мере генерации.
    При `stream=false` возвращает JSON `AskResponse`.
    """
    effective_threshold = body.score_threshold if body.score_threshold is not None else pipeline.score_threshold
    if effective_threshold != pipeline.score_threshold or body.n_results != pipeline.n_results:
        from retrieval.pipeline import RAGPipeline as _P
        pipeline = _P(
            llm=pipeline.llm,
            vector_db_factory=pipeline.vector_db_factory,
            embed_fn=pipeline.embed_fn,
            template=pipeline.template,
            n_results=body.n_results,
            budget=pipeline.budget,
            score_threshold=effective_threshold,
            fallback_answer=pipeline.fallback_answer,
        )

    if body.stream:
        return StreamingResponse(
            _sse_generator(pipeline, body.question, body.collection),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        response = await pipeline.ask(body.question, body.collection)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    return AskResponse(
        answer=response.answer,
        sources=[
            AskSourceItem(
                text=s.text,
                score=s.score,
                doc_id=s.doc_id,
                metadata=s.metadata,
            )
            for s in response.sources
        ],
        confidence=response.confidence,
        latency_ms=response.latency_ms,
    )
