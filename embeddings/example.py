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

from config import settings
from embeddings.adapters import (
    FakeEmbeddingService,
    OpenAIEmbeddingService,
    SentenceTransformersService,
)
from embeddings.cache import EmbeddingCache
from embeddings.ports import EmbeddingService
from embeddings.service import CachedEmbeddingService
from embeddings.utils import cosine_similarity, l2_norm

TEXTS = [
    "RAG объединяет поиск по документам и генерацию ответа.",
    "Vector store хранит embedding-векторы текстовых чанков.",
    "Кэш эмбеддингов снижает число повторных вызовов модели.",
]


def main() -> None:
    """Запустить пример embedding-сервиса на нескольких текстах."""
    args = parse_args()
    base, model_name = build_base_service(args.provider)
    cache = EmbeddingCache(args.cache_path)
    service = CachedEmbeddingService(base=base, cache=cache, model_name=model_name)

    vectors = service.embed_batch(TEXTS)
    assert len(vectors) == len(TEXTS)
    assert all(len(vector) == service.dimension() for vector in vectors)

    repeated_first = service.embed(TEXTS[0])
    assert repeated_first == vectors[0]

    print("Пример embeddings прошёл успешно")
    print(f"Провайдер: {args.provider}")
    print(f"Модель: {model_name}")
    print(f"Количество текстов: {len(TEXTS)}")
    print(f"Размерность вектора: {service.dimension()}")
    print(f"L2-норма первого вектора: {l2_norm(vectors[0]):.6f}")
    print(f"Первые 8 координат первого вектора: {format_vector_head(vectors[0])}")
    print(f"Сходство текста 1 и 2: {cosine_similarity(vectors[0], vectors[1]):.6f}")
    print(f"Сходство текста 1 и 3: {cosine_similarity(vectors[0], vectors[2]):.6f}")
    print(f"Путь к SQLite-кэшу: {args.cache_path}")

    if isinstance(base, FakeEmbeddingService):
        print(f"Вызовы базового сервиса: {base.calls}")


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


def build_base_service(provider: str) -> tuple[EmbeddingService, str]:
    """Создать базовый embedding-сервис для выбранного провайдера."""
    if provider == "fake":
        model_name = "fake-model"
        return FakeEmbeddingService(size=8, model_name=model_name, normalize=True), model_name

    if provider == "openai":
        return (
            OpenAIEmbeddingService(
                model=settings.embedding_model,
                api_key=settings.openai_api_key or None,
                normalize=settings.embedding_normalize,
            ),
            settings.embedding_model,
        )

    model_name = settings.embedding_model
    return (
        SentenceTransformersService(
            model_name=model_name,
            normalize=settings.embedding_normalize,
        ),
        model_name,
    )


def format_vector_head(vector: list[float], limit: int = 8) -> str:
    """Отформатировать первые координаты вектора для вывода в консоль."""
    head = ", ".join(f"{value:.6f}" for value in vector[:limit])
    return f"[{head}]"


if __name__ == "__main__":
    main()
