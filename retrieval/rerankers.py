"""Конкретные реализации Reranker.

CrossEncoderReranker  — точное ранжирование через cross-encoder модель.
MMRReranker           — разнообразная выборка через Maximal Marginal Relevance.
"""

from __future__ import annotations

import math
from typing import Callable

from .pipeline import SourceChunk
from .ports import Reranker


class CrossEncoderReranker(Reranker):
    """Ранжирует кандидатов через cross-encoder: каждую пару (query, doc)
    модель оценивает совместно, что точнее, чем bi-encoder cosine similarity.

    Зависимость: ``sentence-transformers`` (уже в pyproject.toml).
    Первый вызов ``rerank()`` загружает модель с диска/HuggingFace.

    Args:
        model: Имя или путь к cross-encoder модели. По умолчанию
            ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — лёгкая модель
            для ранжирования passage retrieval.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model
        self._model = None  # ленивая загрузка

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[SourceChunk]) -> list[SourceChunk]:
        """Оценить каждую пару (query, doc) и вернуть кандидатов по убыванию score.

        Args:
            query: Исходный запрос пользователя.
            candidates: Список SourceChunk из векторного поиска.

        Returns:
            Пересортированный список SourceChunk. Поле ``score`` заменяется
            нормализованным значением из cross-encoder (sigmoid).
        """
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, c.text) for c in candidates]
        raw_scores: list[float] = model.predict(pairs).tolist()

        # Нормализуем через sigmoid → [0, 1]
        normalized = [_sigmoid(s) for s in raw_scores]

        ranked = sorted(
            zip(candidates, normalized),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            SourceChunk(
                text=chunk.text,
                score=float(score),
                doc_id=chunk.doc_id,
                metadata=chunk.metadata,
            )
            for chunk, score in ranked
        ]


class MMRReranker(Reranker):
    """Ранжирует кандидатов алгоритмом Maximal Marginal Relevance (MMR).

    MMR итеративно выбирает документ, максимизирующий:

        λ · sim(doc, query) − (1 − λ) · max_{s ∈ S} sim(doc, s)

    где S — уже выбранные документы. При λ=1 — чистый relevance (как без MMR);
    при λ=0 — максимальное разнообразие.

    Для вычисления doc-doc сходства необходим ``embed_fn``. Если не передан,
    реранкер деградирует до сортировки по исходному score.

    Args:
        lambda_: Баланс между релевантностью и разнообразием. [0, 1].
        embed_fn: Функция ``text → vector`` для получения векторов документов.
            Если ``None`` — используется только исходный score (без div.).
    """

    def __init__(
        self,
        lambda_: float = 0.5,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ должен быть в [0, 1], получен {lambda_}")
        self.lambda_ = lambda_
        self.embed_fn = embed_fn

    def rerank(self, query: str, candidates: list[SourceChunk]) -> list[SourceChunk]:
        """Выбрать диверсифицированный список кандидатов через MMR.

        Args:
            query: Исходный запрос пользователя.
            candidates: Список SourceChunk из векторного поиска.

        Returns:
            Пересортированный список SourceChunk той же длины, где
            первые элементы максимизируют баланс relevance/diversity.
        """
        if not candidates:
            return []

        if self.embed_fn is None:
            # Нет embed_fn → просто сортируем по score (relevance only)
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        # Вычисляем эмбеддинги документов
        embeddings = [self.embed_fn(c.text) for c in candidates]
        query_emb = self.embed_fn(query)

        # query-doc similarity: пересчитываем через embed_fn для согласованности
        query_sims = [_cosine_sim(query_emb, e) for e in embeddings]

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        while remaining:
            if not selected_indices:
                # Первый элемент — просто наиболее релевантный
                best_idx = max(remaining, key=lambda i: query_sims[i])
            else:
                # MMR score: λ·sim(d, q) − (1−λ)·max_{s} sim(d, s)
                def mmr_score(i: int) -> float:
                    rel = query_sims[i]
                    div = max(
                        _cosine_sim(embeddings[i], embeddings[s])
                        for s in selected_indices
                    )
                    return self.lambda_ * rel - (1.0 - self.lambda_) * div

                best_idx = max(remaining, key=mmr_score)

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Внутренние вспомогательные функции
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Сигмоида: отображает R → (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
