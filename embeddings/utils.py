"""Вспомогательные функции для embedding-векторов."""

import math


def l2_normalize(vector: list[float]) -> list[float]:
    """Вернуть вектор, приведённый к единичной L2-норме.

    Args:
        vector: Исходный embedding-вектор.

    Returns:
        Нормализованный вектор. Если входной вектор нулевой, возвращается его копия.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return list(vector)
    return [value / norm for value in vector]


def normalize_batch(vectors: list[list[float]]) -> list[list[float]]:
    """Нормализовать батч векторов.

    Args:
        vectors: Список embedding-векторов.

    Returns:
        Список нормализованных embedding-векторов.
    """
    return [l2_normalize(vector) for vector in vectors]


def l2_norm(vector: list[float]) -> float:
    """Посчитать L2-норму вектора.

    Args:
        vector: Embedding-вектор.

    Returns:
        Евклидова длина вектора.
    """
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Посчитать cosine similarity двух векторов.

    Args:
        left: Первый embedding-вектор.
        right: Второй embedding-вектор.

    Returns:
        Значение cosine similarity в диапазоне от -1 до 1.

    Raises:
        ValueError: Если векторы имеют разную размерность.
    """
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")

    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)
