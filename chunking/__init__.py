"""Разбивка текста на чанки для индексации в векторном хранилище."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Разбить текст на перекрывающиеся фрагменты.

    Использует RecursiveCharacterTextSplitter из langchain-text-splitters:
    пробует разбить по абзацам → предложениям → словам → символам.

    Args:
        text: Исходный текст.
        chunk_size: Максимальный размер чанка в символах.
        chunk_overlap: Перекрытие между соседними чанками в символах.

    Returns:
        Список непустых текстовых фрагментов.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return [c for c in splitter.split_text(text) if c.strip()]
