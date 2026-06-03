"""Pydantic-схемы для всех запросов и ответов API."""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    """Состояние одного компонента системы."""

    status: Literal["ok", "error", "not_configured"]
    detail: str | None = None


class HealthResponse(BaseModel):
    """Ответ эндпоинта GET /v1/health."""

    # overall: "ok" — все компоненты ok; "degraded" — хотя бы один error/not_configured.
    status: Literal["ok", "degraded"]
    components: dict[str, ComponentHealth]


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class CollectionItem(BaseModel):
    """Информация об одной коллекции векторного хранилища."""

    name: str
    vectors_count: int = Field(ge=0)


class CollectionsResponse(BaseModel):
    """Ответ эндпоинта GET /v1/collections."""

    collections: list[CollectionItem]


class DeleteCollectionResponse(BaseModel):
    """Ответ эндпоинта DELETE /v1/collections/{name}."""

    deleted: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    """Стандартное тело ошибки (используется в responses= у роутеров)."""

    detail: str
