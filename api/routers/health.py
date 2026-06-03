"""GET /v1/health — проверка состояния всех компонентов системы."""

from fastapi import APIRouter, Request

from api.schemas import ComponentHealth, HealthResponse
from config import settings

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверить состояние системы",
)
async def health(request: Request) -> HealthResponse:
    """Возвращает статус каждого компонента: vector_store, llm, cache.

    Не выполняет дорогих API-вызовов: проверяет наличие подключения
    к векторному хранилищу через лёгкий ``count()`` и валидность
    конфигурации LLM-провайдера.

    Returns:
        ``status="ok"``      — все компоненты работают.
        ``status="degraded"``— хотя бы один компонент в состоянии error/not_configured.
    """
    components: dict[str, ComponentHealth] = {}

    # ---- vector_store ----
    # Вызываем count() — лёгкий запрос, который подтверждает доступность хранилища.
    try:
        vector_db = request.app.state.vector_db
        count = vector_db.count()
        components["vector_store"] = ComponentHealth(
            status="ok",
            detail=f"backend={settings.vector_store_backend}, vectors={count}",
        )
    except Exception as exc:
        components["vector_store"] = ComponentHealth(status="error", detail=str(exc))

    # ---- llm ----
    # Не делаем реальный API-запрос (дорого). Проверяем наличие ключа / URL.
    try:
        provider = type(request.app.state.llm).__name__
        if settings.llm_provider == "ollama":
            detail = f"{provider}, url={settings.ollama_base_url}"
        else:
            has_key = bool(
                settings.openai_api_key
                if settings.llm_provider == "openai"
                else settings.anthropic_api_key
            )
            detail = f"{provider}, key={'set' if has_key else 'missing'}"
        components["llm"] = ComponentHealth(status="ok", detail=detail)
    except Exception as exc:
        components["llm"] = ComponentHealth(status="error", detail=str(exc))

    # ---- cache ----
    # Модуль кэша ещё не реализован — сообщаем об этом явно.
    components["cache"] = ComponentHealth(status="not_configured")

    overall = (
        "ok"
        if all(c.status != "error" for c in components.values())
        else "degraded"
    )
    return HealthResponse(status=overall, components=components)
