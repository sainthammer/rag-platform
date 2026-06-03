"""GET /v1/collections и DELETE /v1/collections/{name}.

Для операций уровня коллекции (список всех, удаление) нужен доступ
к клиенту хранилища напрямую, а не через per-collection VectorDB.
Клиент хранится в ``app.state.store_client``; тип бэкенда — в ``app.state.store_backend``.
"""

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas import CollectionItem, CollectionsResponse, DeleteCollectionResponse

router = APIRouter(tags=["collections"])


def _list_collections(request: Request) -> list[CollectionItem]:
    """Вернуть список коллекций из текущего бэкенда."""
    client = request.app.state.store_client
    backend: str = request.app.state.store_backend

    if backend == "chroma":
        # list_collections() возвращает список объектов Collection.
        raw = client.list_collections()
        return [CollectionItem(name=col.name, vectors_count=col.count()) for col in raw]

    # qdrant: get_collections() → CollectionsResponse с .collections: list[CollectionDescription]
    raw = client.get_collections().collections
    items = []
    for col in raw:
        try:
            count = client.count(col.name).count
        except Exception:
            count = 0
        items.append(CollectionItem(name=col.name, vectors_count=count))
    return items


def _delete_collection(request: Request, name: str) -> None:
    """Удалить коллекцию из текущего бэкенда."""
    client = request.app.state.store_client
    backend: str = request.app.state.store_backend

    if backend == "chroma":
        client.delete_collection(name)
    else:
        client.delete_collection(name)


@router.get(
    "/collections",
    response_model=CollectionsResponse,
    summary="Список всех коллекций",
)
async def list_collections(request: Request) -> CollectionsResponse:
    """Вернуть все коллекции векторного хранилища с числом векторов в каждой."""
    try:
        items = _list_collections(request)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return CollectionsResponse(collections=items)


@router.delete(
    "/collections/{name}",
    response_model=DeleteCollectionResponse,
    summary="Удалить коллекцию",
    responses={404: {"description": "Коллекция не найдена"}},
)
async def delete_collection(name: str, request: Request) -> DeleteCollectionResponse:
    """Безвозвратно удалить коллекцию и все её векторы.

    Args:
        name: Имя коллекции.
    """
    # Сначала проверяем, что коллекция существует.
    items = _list_collections(request)
    if not any(c.name == name for c in items):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Коллекция {name!r} не найдена",
        )
    try:
        _delete_collection(request, name)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return DeleteCollectionResponse(deleted=name)
