from config import Settings
from embeddings.adapters import FakeEmbeddingService
from embeddings.cache import EmbeddingCache, cache_key
from embeddings.service import (
    CachedEmbeddingService,
    build_base_embedding_service,
    build_document_embedding_service,
    build_query_embedding_service,
)


def test_cache_key_uses_text_and_model_name() -> None:
    assert cache_key("hello", "model-a") == cache_key("hello", "model-a")
    assert cache_key("hello", "model-a") != cache_key("hello", "model-b")


def test_cache_roundtrip() -> None:
    cache = EmbeddingCache(":memory:")
    vector = [0.1, 0.2, 0.3]

    cache.set("hello", "model-a", vector)

    assert cache.get("hello", "model-a") == vector
    assert cache.get("hello", "model-b") is None


def test_get_many_preserves_order_missing_values_and_duplicates() -> None:
    cache = EmbeddingCache(":memory:")
    first = [0.1, 0.2]
    second = [0.3, 0.4]
    cache.set_many(["first", "second"], "model-a", [first, second])

    result = cache.get_many(["second", "missing", "first", "second"], "model-a")

    assert result == [second, None, first, second]


def test_cached_service_hits_cache_on_repeated_text() -> None:
    base = FakeEmbeddingService(size=4)
    cache = EmbeddingCache(":memory:")
    service = CachedEmbeddingService(base=base, cache=cache, model_name="fake-model")

    first = service.embed("same text")
    second = service.embed("same text")

    assert second == first
    assert base.calls == [["same text"]]


def test_cached_service_only_embeds_missing_texts() -> None:
    base = FakeEmbeddingService(size=4)
    cache = EmbeddingCache(":memory:")
    service = CachedEmbeddingService(base=base, cache=cache, model_name="fake-model")

    service.embed_batch(["cached", "missing"])
    base.calls.clear()
    service.embed_batch(["cached", "new"])

    assert base.calls == [["new"]]


def test_cached_service_embeds_duplicate_missing_text_once() -> None:
    base = FakeEmbeddingService(size=4)
    cache = EmbeddingCache(":memory:")
    service = CachedEmbeddingService(base=base, cache=cache, model_name="fake-model")

    vectors = service.embed_batch(["same", "same", "other"])

    assert vectors[0] == vectors[1]
    assert base.calls == [["same", "other"]]


def test_query_embedding_factory_uses_cache_when_enabled(monkeypatch) -> None:
    fake = FakeEmbeddingService(size=4)
    monkeypatch.setattr(
        "embeddings.service.OpenAIEmbeddingService",
        lambda **_: fake,
    )
    settings = Settings(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="fake-model",
        EMBEDDING_CACHE_ENABLED=True,
        EMBEDDING_CACHE_PATH=":memory:",
    )

    service = build_query_embedding_service(settings)

    assert isinstance(service, CachedEmbeddingService)
    assert service.base is fake


def test_base_embedding_factory_skips_cache_when_enabled(monkeypatch) -> None:
    fake = FakeEmbeddingService(size=4)
    monkeypatch.setattr(
        "embeddings.service.OpenAIEmbeddingService",
        lambda **_: fake,
    )
    settings = Settings(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="fake-model",
        EMBEDDING_CACHE_ENABLED=True,
        EMBEDDING_CACHE_PATH=":memory:",
    )

    service = build_base_embedding_service(settings)

    assert service is fake


def test_document_embedding_factory_skips_cache_when_enabled(monkeypatch) -> None:
    fake = FakeEmbeddingService(size=4)
    monkeypatch.setattr(
        "embeddings.service.OpenAIEmbeddingService",
        lambda **_: fake,
    )
    settings = Settings(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="fake-model",
        EMBEDDING_CACHE_ENABLED=True,
        EMBEDDING_CACHE_PATH=":memory:",
    )

    service = build_document_embedding_service(settings)

    assert service is fake
