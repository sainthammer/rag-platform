from embeddings.adapters import FakeEmbeddingService
from embeddings.cache import EmbeddingCache, cache_key
from embeddings.service import CachedEmbeddingService


def test_cache_key_uses_text_and_model_name() -> None:
    assert cache_key("hello", "model-a") == cache_key("hello", "model-a")
    assert cache_key("hello", "model-a") != cache_key("hello", "model-b")


def test_cache_roundtrip() -> None:
    cache = EmbeddingCache(":memory:")
    vector = [0.1, 0.2, 0.3]

    cache.set("hello", "model-a", vector)

    assert cache.get("hello", "model-a") == vector
    assert cache.get("hello", "model-b") is None


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
