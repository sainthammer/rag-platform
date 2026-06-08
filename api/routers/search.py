"""POST /v1/search — семантический поиск по векторному хранилищу."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.deps import get_embed_service
from api.schemas import SearchRequest, SearchResponse, SearchResultItem
from config import build_vector_db, settings
from embeddings.ports import EmbeddingService

router = APIRouter(tags=["search"])

_SCORE_DENOM = 1.0  # distance → score: 1 / (1 + distance)


def _distance_to_score(d: float) -> float:
    return 1.0 / (1.0 + d)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Семантический поиск по коллекции",
)
async def search(
    body: SearchRequest,
    embed_service: Annotated[EmbeddingService, Depends(get_embed_service)],
) -> SearchResponse:
    """Векторизовать запрос и вернуть n_results ближайших чанков.

    Результаты с `score < score_threshold` отфильтровываются.
    """
    embedding = embed_service.embed(body.query)

    try:
        db = build_vector_db(settings, collection=body.collection)
        raw = db.search(embedding, n_results=body.n_results)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    items = [
        SearchResultItem(
            id=doc_id,
            text=doc,
            score=_distance_to_score(dist),
            metadata=meta,
        )
        for doc_id, doc, dist, meta in zip(
            raw.ids, raw.documents, raw.distances, raw.metadatas
        )
        if _distance_to_score(dist) >= body.score_threshold
    ]

    return SearchResponse(results=items, query=body.query, collection=body.collection)
