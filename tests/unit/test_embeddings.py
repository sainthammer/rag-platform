import pytest

from embeddings.adapters import FakeEmbeddingService
from embeddings.utils import cosine_similarity, l2_norm


def test_dimension_returns_configured_size() -> None:
    service = FakeEmbeddingService(size=12)

    assert service.dimension() == 12


def test_normalize_scales_vectors_to_unit_norm() -> None:
    service = FakeEmbeddingService(size=8, normalize=True)

    vector = service.embed("hello")

    assert l2_norm(vector) == pytest.approx(1.0)


def test_batch_matches_single_embeddings() -> None:
    service = FakeEmbeddingService(size=8)
    texts = ["alpha", "beta", "gamma"]

    batch = service.embed_batch(texts)
    singles = [service.embed(text) for text in texts]

    assert batch == singles
    assert len(batch) == len(texts)


def test_cosine_similarity_is_highest_for_identical_vector() -> None:
    service = FakeEmbeddingService(size=8, normalize=True)
    first = service.embed("same text")
    same = service.embed("same text")
    different = service.embed("different text")

    assert cosine_similarity(first, same) == pytest.approx(1.0)
    assert cosine_similarity(first, same) > cosine_similarity(first, different)
