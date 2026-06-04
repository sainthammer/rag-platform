"""SQLite-кэш для embedding-векторов.

Кэш хранит результат векторизации по ключу
``sha256(text + model_name)``. Это позволяет повторно использовать
эмбеддинги и не делать лишние вызовы внешних моделей.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def cache_key(text: str, model_name: str) -> str:
    """Собрать ключ кэша: ``sha256(text + model_name)``.

    Args:
        text: Текст, для которого строится embedding.
        model_name: Имя embedding-модели.

    Returns:
        Hex-строка SHA256, используемая как первичный ключ в SQLite.
    """
    return hashlib.sha256(f"{text}{model_name}".encode()).hexdigest()


class EmbeddingCache:
    """Персистентный SQLite-кэш для embedding-векторов."""

    def __init__(self, path: str | Path = ".cache/embeddings.sqlite3") -> None:
        """Создать кэш и инициализировать SQLite-таблицу.

        Args:
            path: Путь к SQLite-файлу или ``":memory:"`` для in-memory базы.
        """
        self._memory_conn: sqlite3.Connection | None = None
        if str(path) == ":memory:":
            self.path = Path(":memory:")
            self._memory_conn = sqlite3.connect(":memory:")
            self._init_db()
            return

        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get(self, text: str, model_name: str) -> list[float] | None:
        """Вернуть сохранённый вектор.

        Если значения нет в кэше, возвращается ``None``.

        Args:
            text: Исходный текст.
            model_name: Имя embedding-модели.

        Returns:
            Сохранённый embedding-вектор или ``None``.
        """
        key = cache_key(text, model_name)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector_json FROM embeddings WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return [float(value) for value in json.loads(row[0])]

    def set(self, text: str, model_name: str, vector: list[float]) -> None:
        """Сохранить embedding-вектор в кэш.

        Args:
            text: Исходный текст.
            model_name: Имя embedding-модели.
            vector: Embedding-вектор для сохранения.
        """
        key = cache_key(text, model_name)
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        vector_json = json.dumps(vector, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (key, model_name, text_hash, vector_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    model_name = excluded.model_name,
                    text_hash = excluded.text_hash,
                    vector_json = excluded.vector_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (key, model_name, text_hash, vector_json),
            )

    def get_many(self, texts: list[str], model_name: str) -> list[list[float] | None]:
        """Вернуть кэшированные векторы в порядке входных текстов.

        Args:
            texts: Список исходных текстов.
            model_name: Имя embedding-модели.

        Returns:
            Список той же длины, где отсутствующие значения представлены ``None``.
        """
        return [self.get(text, model_name) for text in texts]

    def set_many(self, texts: list[str], model_name: str, vectors: list[list[float]]) -> None:
        """Сохранить батч embedding-векторов одной транзакцией.

        Args:
            texts: Список исходных текстов.
            model_name: Имя embedding-модели.
            vectors: Список embedding-векторов.

        Raises:
            ValueError: Если количество текстов и векторов не совпадает.
        """
        if len(texts) != len(vectors):
            raise ValueError("Тексты и вектора должны иметь одинаковую длину")

        rows = [
            (
                cache_key(text, model_name),
                model_name,
                hashlib.sha256(text.encode()).hexdigest(),
                json.dumps(vector, separators=(",", ":")),
            )
            for text, vector in zip(texts, vectors, strict=True)
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (key, model_name, text_hash, vector_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    model_name = excluded.model_name,
                    text_hash = excluded.text_hash,
                    vector_json = excluded.vector_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                rows,
            )

    def _connect(self) -> sqlite3.Connection:
        """Открыть соединение с SQLite или вернуть in-memory соединение.

        Returns:
            Активное соединение SQLite.
        """
        if self._memory_conn is not None:
            return self._memory_conn
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        """Создать таблицу embeddings, если она ещё не существует."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def cached(fn: Callable[P, T]) -> Callable[P, T]:
    """Декоратор кэширования.

    Ожидает, что у объекта есть атрибуты ``cache`` и ``model_name``.

    Args:
        fn: Метод, который строит embedding для одного текста.

    Returns:
        Обёрнутый метод с проверкой SQLite-кэша перед вычислением.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        """Выполнить метод с проверкой single-text embedding в кэше.

        Args:
            *args: Позиционные аргументы исходного метода.
            **kwargs: Именованные аргументы исходного метода.

        Returns:
            Значение из кэша или результат исходного метода.
        """
        if len(args) < 2 or not isinstance(args[1], str):
            return fn(*args, **kwargs)

        instance = args[0]
        text = args[1]
        cache = getattr(instance, "cache", None)
        model_name = getattr(instance, "model_name", None)
        if cache is None or model_name is None:
            return fn(*args, **kwargs)

        cached_vector = cache.get(text, model_name)
        if cached_vector is not None:
            return cached_vector  # type: ignore[return-value]

        vector = fn(*args, **kwargs)
        cache.set(text, model_name, vector)
        return vector

    return wrapper
