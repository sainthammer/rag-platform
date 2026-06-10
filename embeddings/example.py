"""Пример запуска модуля embeddings.

Запуск:
    python -m embeddings.example
    python -m embeddings.example fake
    python -m embeddings.example openai
    python -m embeddings.example sentence-transformers

По умолчанию используется FakeEmbeddingService, поэтому базовая проверка
не требует API-ключей, скачивания моделей или сетевого доступа.
"""

import argparse

from config import Settings
from embeddings.adapters import FakeEmbeddingService
from embeddings.cache import EmbeddingCache
from embeddings.ports import EmbeddingService
from embeddings.service import (
    CachedEmbeddingService,
    build_document_embedding_service,
    build_query_embedding_service,
)
from embeddings.utils import cosine_similarity, l2_norm

QUERY_TEXT = "Что такое RAG?"
DOCUMENT_TEXTS = [
    "RAG объединяет поиск по документам и генерацию ответа.",
    "Vector store хранит embedding-векторы текстовых чанков.",
    "Кэш эмбеддингов снижает число повторных вызовов модели.",
]


def main() -> None:
    """Запустить пример embedding-сервиса на нескольких текстах."""
    args = parse_args()
    query_service, document_service, model_name, fake_bases = build_example_services(
        provider=args.provider,
        cache_path=args.cache_path,
    )

    query_vector = query_service.embed(QUERY_TEXT)
    document_vectors = document_service.embed_batch(DOCUMENT_TEXTS)
    assert len(document_vectors) == len(DOCUMENT_TEXTS)
    assert len(query_vector) == query_service.dimension()
    assert all(len(vector) == document_service.dimension() for vector in document_vectors)

    repeated_query = query_service.embed(QUERY_TEXT)
    assert repeated_query == query_vector

    print("Пример embeddings прошёл успешно")
    print(f"Провайдер: {args.provider}")
    print(f"Модель: {model_name}")
    print(f"Количество чанков: {len(DOCUMENT_TEXTS)}")
    print(f"Размерность query-вектора: {query_service.dimension()}")
    print(f"Размерность document-вектора: {document_service.dimension()}")
    print(f"L2-норма query-вектора: {l2_norm(query_vector):.6f}")
    print(f"Первые 8 координат query-вектора: {format_vector_head(query_vector)}")
    print(
        "Сходство query и чанка 1: "
        f"{cosine_similarity(query_vector, document_vectors[0]):.6f}"
    )
    print(
        "Сходство query и чанка 2: "
        f"{cosine_similarity(query_vector, document_vectors[1]):.6f}"
    )
    print(f"Путь к SQLite-кэшу query-сервиса: {args.cache_path}")

    for name, service in fake_bases:
        print(f"Вызовы fake-сервиса ({name}): {service.calls}")


def parse_args() -> argparse.Namespace:
    """Прочитать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Проверка embedding-сервиса")
    parser.add_argument(
        "provider",
        nargs="?",
        choices=["fake", "openai", "sentence-transformers"],
        default="fake",
        help="Провайдер эмбеддингов для проверки",
    )
    parser.add_argument(
        "--cache-path",
        default=":memory:",
        help="Путь к SQLite-кэшу. По умолчанию используется in-memory SQLite",
    )
    return parser.parse_args()


def build_example_services(
    provider: str,
    cache_path: str,
) -> tuple[EmbeddingService, EmbeddingService, str, list[tuple[str, FakeEmbeddingService]]]:
    """Создать query/document embedding-сервисы для примера.

    Args:
        provider: Имя провайдера из аргументов CLI.
        cache_path: Путь к SQLite-кэшу query-сервиса.

    Returns:
        Query-сервис, document-сервис, имя модели и fake-сервисы для диагностики.
    """
    if provider == "fake":
        model_name = "fake-model"
        query_base = FakeEmbeddingService(size=8, model_name=model_name, normalize=True)
        document_base = FakeEmbeddingService(size=8, model_name=model_name, normalize=True)
        query_service = CachedEmbeddingService(
            base=query_base,
            cache=EmbeddingCache(cache_path),
            model_name=model_name,
        )
        return (
            query_service,
            document_base,
            model_name,
            [("query", query_base), ("document", document_base)],
        )

    example_settings = Settings(
        EMBEDDING_PROVIDER=provider,
        EMBEDDING_CACHE_PATH=cache_path,
    )
    return (
        build_query_embedding_service(example_settings),
        build_document_embedding_service(example_settings),
        example_settings.embedding_model,
        [],
    )


def format_vector_head(vector: list[float], limit: int = 8) -> str:
    """Отформатировать первые координаты вектора для вывода в консоль."""
    head = ", ".join(f"{value:.6f}" for value in vector[:limit])
    return f"[{head}]"


if __name__ == "__main__":
    main()
