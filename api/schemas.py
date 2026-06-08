"""Pydantic-схемы для всех запросов и ответов API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    status: Literal["ok", "error", "not_configured"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, ComponentHealth]


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class CollectionItem(BaseModel):
    name: str
    vectors_count: int = Field(ge=0)


class CollectionsResponse(BaseModel):
    collections: list[CollectionItem]


class DeleteCollectionResponse(BaseModel):
    deleted: str


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestJobResponse(BaseModel):
    """Ответ POST /v1/ingest — задача поставлена в очередь."""

    job_id: str
    status: Literal["pending", "running", "done", "error"]


class IngestStatusResponse(BaseModel):
    """Ответ GET /v1/ingest/{job_id}."""

    job_id: str
    status: Literal["pending", "running", "done", "error"]
    chunks_indexed: int = 0
    collection: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    collection: str = "default"
    n_results: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str
    collection: str


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    collection: str = "default"
    stream: bool = False
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    n_results: int = Field(default=5, ge=1, le=50)


class AskSourceItem(BaseModel):
    text: str
    score: float
    doc_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSourceItem]
    confidence: float
    latency_ms: float


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

class EvalRunRequest(BaseModel):
    mode: Literal["mock", "ollama"] = "mock"
    max_cases: int = Field(default=5, ge=1, le=46)
    output_path: str = "eval_report.html"


class EvalJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]


class EvalStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    report_path: str | None = None
    hallucination_pass: int | None = None
    hallucination_total: int | None = None
    ragas_avg: dict[str, float | None] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    detail: str
