"""
Скрипт переиндексации (reindex) для векторных БД.

Что делает:
  1. Читает все записи из source порциями.
  2. Пересчитывает эмбеддинги по документам через embed_fn.
  3. Записывает в target (upsert: совпадающие id перезаписываются).
  4. Удаляет из target старые записи, которых нет в source.

Когда нужно:
  - смена модели эмбеддингов
  - миграция между БД (Chroma -> Qdrant и наоборот)
"""

from collections.abc import Callable, Iterator

from vector_store.adapters import ChromaDB, HybridVectorStore, QdrantDB
from vector_store.ports import VectorDB
from vector_store.utils import to_uuid

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm_installed = None

Batch = tuple[list[str], list[str], list[dict]]
EmbedFn = Callable[[list[str]], list[list[float]]]


def get_real_store(store: VectorDB) -> VectorDB:
    """HybridVectorStore — это обёртка. Достаём из неё настоящий QdrantDB."""
    if isinstance(store, HybridVectorStore):
        return store.store
    return store


def read_batches(store: VectorDB, batch_size: int = 256) -> Iterator[Batch]:
    """
    Читает все записи из стора порциями.
    На каждой итерации отдаёт три списка: (ids, documents, metadatas).
    """
    store = get_real_store(store)

    if isinstance(store, ChromaDB):
        offset = 0
        while True:
            raw = store.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = raw["ids"]
            if len(ids) == 0:
                break

            documents = [doc if doc else "" for doc in raw["documents"] or []]
            metadatas = [dict(meta) if meta else {} for meta in raw["metadatas"] or []]

            yield ids, documents, metadatas
            offset = offset + len(ids)

    elif isinstance(store, QdrantDB):
        offset = None
        while True:
            points, offset = store.client.scroll(
                collection_name=store.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if len(points) == 0:
                break

            payloads = [point.payload or {} for point in points]

            ids = [str(point.id) for point in points]
            documents = [payload.get("document", "") for payload in payloads]
            metadatas = [
                {key: value for key, value in payload.items() if key != "document"}
                for payload in payloads
            ]

            yield ids, documents, metadatas

            if offset is None:
                break

    else:
        raise TypeError("Неподдерживаемый тип стора: " + type(store).__name__)


def delete_old_records(target: VectorDB, keep_ids: set[str], batch_size: int = 256) -> int:
    """
    Удаляет из target записи, чьих id нет в keep_ids.
    keep_ids — множество id в UUID-виде (после to_uuid).
    Возвращает количество удалённых записей.
    """
    old_ids = [
        id
        for ids, documents, metadatas in read_batches(target, batch_size)
        for id in ids
        if to_uuid(id) not in keep_ids
    ]

    for i in range(0, len(old_ids), batch_size):
        target.delete(old_ids[i : i + batch_size])

    return len(old_ids)


def reindex(
    source: VectorDB,
    target: VectorDB,
    embed_fn: EmbedFn,
    batch_size: int = 256,
    delete_old: bool = True,
    show_progress: bool = True,
) -> int:
    """
    Переиндексация: source -> target.

    source и target могут быть одним и тем же стором —
    тогда это пересчёт векторов "на месте" (upsert по тем же id).

    args:
        source: откуда читаем (ChromaDB / QdrantDB / HybridVectorStore)
        target: куда пишем
        embed_fn: функция, которая принимает список текстов
            и возвращает список эмбеддингов
        batch_size: размер порции чтения/записи
        delete_old: удалить из target записи, которых нет в source.
            Удаление происходит ПОСЛЕ успешной записи всех новых данных,
            поэтому при падении посередине старые данные не теряются.
        show_progress: показывать прогресс-бар

    returns:
        количество переиндексированных записей
    """
    total = source.count()
    processed = 0
    new_ids: set[str] = set()

    progress_bar = None
    if show_progress and tqdm_installed is not None:
        progress_bar = tqdm(total=total, desc="reindex", unit="doc")

    for ids, documents, metadatas in read_batches(source, batch_size):
        embeddings = embed_fn(documents)

        if len(embeddings) != len(ids):
            raise ValueError("embed_fn вернула не столько векторов, сколько было документов")

        target.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

        new_ids.update(to_uuid(id) for id in ids)

        processed = processed + len(ids)
        if progress_bar is not None:
            progress_bar.update(len(ids))
        elif show_progress:
            print("reindex:", processed, "/", total)

    if progress_bar is not None:
        progress_bar.close()

    if delete_old:
        deleted = delete_old_records(target, new_ids, batch_size)
        if show_progress and deleted > 0:
            print("Удалено старых записей:", deleted)

    return processed
