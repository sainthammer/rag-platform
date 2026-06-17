"""Вспомогательные функции для внутренней логики чанкинга."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(text: str) -> str:
    """Нормализовать переносы строк и горизонтальные пробелы.

    Args:
        text: Исходный текст.

    Returns:
        Текст с переносами строк в формате ``\n`` и очищенными пробелами.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def stable_hash(*parts: object, length: int = 16) -> str:
    """Построить детерминированный SHA-256 hash заданной длины.

    Args:
        *parts: Части, из которых строится hash.
        length: Длина возвращаемой hex-строки.

    Returns:
        Стабильная hex-строка, пригодная для id и дедупликации.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:length]


def read_text_file(path: Path) -> str:
    """Прочитать текстовый файл как UTF-8 с поддержкой BOM.

    Args:
        path: Путь к файлу.

    Returns:
        Содержимое файла.

    Raises:
        OSError: Если файл нельзя открыть или прочитать.
        UnicodeDecodeError: Если содержимое не декодируется как UTF-8.
    """
    return path.read_text(encoding="utf-8-sig")


def sliding_windows(text: str, size: int, overlap: int) -> Iterable[tuple[int, str]]:
    """Вернуть последовательность символьных окон с overlap.

    Args:
        text: Текст для разбиения.
        size: Максимальный размер окна в символах.
        overlap: Количество символов, повторяемых между соседними окнами.

    Returns:
        Итератор кортежей ``(start, piece)``, где ``start`` - позиция окна.

    Raises:
        ValueError: Если ``size`` меньше или равен нулю, ``overlap`` отрицательный
            или ``overlap`` не меньше ``size``.
    """
    if size <= 0:
        raise ValueError("chunk_size должен быть больше 0")
    if overlap < 0:
        raise ValueError("overlap не может быть отрицательным")
    if overlap >= size:
        raise ValueError("overlap должен быть меньше chunk_size")

    start = 0
    text_length = len(text)
    step = size - overlap
    while start < text_length:
        end = min(start + size, text_length)
        piece = text[start:end].strip()
        if piece:
            yield start, piece
        if end == text_length:
            break
        start += step
