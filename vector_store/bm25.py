"""BM25-векторизатор для разреженного индекса Qdrant.

Формирует разреженные векторы (индексы + веса), совместимые с
``qdrant_client.http.models.SparseVector``.

Типичный сценарий использования::

    vectorizer = BM25SparseVectorizer()
    vectorizer.fit(corpus)          # вычисление IDF по корпусу
    sv = vectorizer.transform(text) # индексы/веса для документа или запроса
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class SparseVector:
    """Разреженный вектор в виде параллельных массивов индексов термов и BM25-весов."""

    indices: list[int]
    values: list[float]


class BM25SparseVectorizer:
    """BM25-векторизатор: преобразует сырой текст в ``SparseVector`` для Qdrant.

    Args:
        k1: Параметр насыщения частоты термина (рекомендуемый диапазон: 1.2–1.5).
        b:  Параметр нормализации по длине документа.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}
        self._idf: dict[int, float] = {}
        self._avgdl: float = 0.0

    @property
    def fitted(self) -> bool:
        """``True``, если векторизатор уже обучен на корпусе."""
        return bool(self._vocab)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Латиница + кириллица + цифры
        return re.findall(r"[a-zа-яёa-z0-9]+", text.lower())

    def fit(self, corpus: list[str]) -> "BM25SparseVectorizer":
        """Построить словарь и вычислить IDF-веса по *corpus*.

        Повторный вызов полностью заменяет существующую модель.
        """
        n = len(corpus)
        if n == 0:
            return self

        tokenized = [self._tokenize(doc) for doc in corpus]

        # Словарь: токен → целочисленный индекс
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab

        # Документная частота по индексам словаря
        df: Counter[int] = Counter()
        for tokens in tokenized:
            for token in set(tokens):
                if token in vocab:
                    df[vocab[token]] += 1

        # BM25+ IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {
            idx: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for idx, freq in df.items()
        }

        self._avgdl = sum(len(t) for t in tokenized) / n
        return self

    def transform(self, text: str) -> SparseVector:
        """Вычислить BM25-разреженный вектор для *text* по обученному словарю.

        Токены, отсутствующие в словаре, молча игнорируются.
        Возвращает пустой ``SparseVector``, если векторизатор не обучен
        или ни один токен словаря не встретился в *text*.
        """
        if not self._vocab:
            return SparseVector(indices=[], values=[])

        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        dl = len(tokens)
        tf = Counter(tokens)
        avgdl = max(self._avgdl, 1.0)

        indices: list[int] = []
        values: list[float] = []
        for token, count in tf.items():
            idx = self._vocab.get(token)
            if idx is None:
                continue
            idf = self._idf.get(idx, 0.0)
            if idf <= 0.0:
                continue
            tf_norm = (count * (self.k1 + 1)) / (
                count + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            )
            indices.append(idx)
            values.append(idf * tf_norm)

        return SparseVector(indices=indices, values=values)
