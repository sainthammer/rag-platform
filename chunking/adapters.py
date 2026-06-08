"""Конкретные реализации Chunker: фиксированный, по заголовкам, семантический.

Все чанкеры реализуют интерфейс ``Chunker.split(text, metadata)``
и возвращают ``list[Chunk]`` с уникальными id и заполненными метаданными.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from .ports import Chunk, Chunker

# Паттерн для Markdown-заголовков: # Title, ## Title, ...
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class FixedSizeChunker(Chunker):
    """Разбивает текст на фрагменты фиксированного размера с перекрытием.

    Использует ``RecursiveCharacterTextSplitter`` из langchain-text-splitters:
    приоритет разбивки по абзацам → предложениям → словам → символам.

    Args:
        chunk_size: Максимальный размер чанка в символах.
        chunk_overlap: Перекрытие соседних чанков в символах.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Разбить текст методом скользящего окна.

        Args:
            text: Исходный текст.
            metadata: Базовые метаданные для каждого чанка.

        Returns:
            Список чанков с полем ``strategy="fixed"`` в метаданных.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        parts = [p for p in splitter.split_text(text) if p.strip()]
        base_meta = dict(metadata or {})
        return [
            Chunk(
                text=part,
                metadata={**base_meta, "chunk_index": i, "strategy": "fixed"},
                id=_make_id(part, i),
            )
            for i, part in enumerate(parts)
        ]


class ByHeaderChunker(Chunker):
    """Разбивает Markdown-текст на секции по заголовкам (#, ##, ###...).

    Каждый заголовок образует новую секцию. Если секция длиннее
    ``chunk_size`` символов, она дополнительно дробится через
    ``FixedSizeChunker``. Текст до первого заголовка также становится чанком.

    Args:
        chunk_size: Максимальный размер одной секции в символах.
        chunk_overlap: Перекрытие при дополнительном дроблении длинных секций.
        min_chunk_size: Секции короче этого порога пропускаются.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        min_chunk_size: int = 30,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Разбить текст по заголовкам Markdown.

        Args:
            text: Текст (обычно Markdown).
            metadata: Базовые метаданные для каждого чанка.

        Returns:
            Список чанков. Поля метаданных: ``strategy="by_header"``,
            ``header`` — текст заголовка (пустая строка для преамбулы).
        """
        sections = _split_by_headers(text)
        base_meta = dict(metadata or {})
        result: list[Chunk] = []

        for header, body in sections:
            section_text = (f"{header}\n{body}" if header else body).strip()
            if not section_text or len(section_text) < self.min_chunk_size:
                continue

            if len(section_text) <= self.chunk_size:
                idx = len(result)
                result.append(Chunk(
                    text=section_text,
                    metadata={
                        **base_meta,
                        "chunk_index": idx,
                        "header": header,
                        "strategy": "by_header",
                    },
                    id=_make_id(section_text, idx),
                ))
            else:
                # Секция слишком большая — дополнительно дробим
                sub_chunker = FixedSizeChunker(self.chunk_size, self.chunk_overlap)
                for sub in sub_chunker.split(section_text, metadata):
                    idx = len(result)
                    result.append(Chunk(
                        text=sub.text,
                        metadata={
                            **base_meta,
                            **sub.metadata,
                            "chunk_index": idx,
                            "header": header,
                            "strategy": "by_header",
                        },
                        id=_make_id(sub.text, idx),
                    ))

        return result


class SemanticChunker(Chunker):
    """Объединяет параграфы в чанки по смысловой близости или размеру.

    Без ``embed_fn``: параграфы (разделённые пустой строкой) объединяются
    жадно, пока суммарный размер не превысит ``chunk_size``.

    С ``embed_fn``: соседние параграфы объединяются, если их косинусное
    сходство >= ``similarity_threshold`` и суммарный размер <= ``chunk_size``.

    Args:
        chunk_size: Мягкий предел размера чанка в символах.
        similarity_threshold: Порог косинусного сходства для объединения
            (используется только при передаче ``embed_fn``).
        embed_fn: Функция векторизации текста. Если ``None`` — используется
            разбивка по размеру без вычисления эмбеддингов.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        similarity_threshold: float = 0.85,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.embed_fn = embed_fn

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Разбить текст на семантически однородные чанки.

        Args:
            text: Исходный текст.
            metadata: Базовые метаданные для каждого чанка.

        Returns:
            Список чанков с ``strategy="semantic"`` в метаданных.
        """
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paragraphs:
            return []

        base_meta = dict(metadata or {})

        if self.embed_fn is not None:
            groups = self._semantic_groups(paragraphs)
        else:
            groups = self._size_groups(paragraphs)

        return [
            Chunk(
                text=group,
                metadata={**base_meta, "chunk_index": i, "strategy": "semantic"},
                id=_make_id(group, i),
            )
            for i, group in enumerate(groups)
        ]

    def _size_groups(self, paragraphs: list[str]) -> list[str]:
        """Объединить параграфы жадно, не превышая chunk_size."""
        groups: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            added_len = len(para) + (2 if current else 0)
            if current and current_len + added_len > self.chunk_size:
                groups.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += added_len

        if current:
            groups.append("\n\n".join(current))

        return groups

    def _semantic_groups(self, paragraphs: list[str]) -> list[str]:
        """Объединить соседние параграфы по косинусному сходству эмбеддингов."""
        assert self.embed_fn is not None
        embeddings = [self.embed_fn(p) for p in paragraphs]
        groups: list[str] = []
        current: list[str] = [paragraphs[0]]
        current_len = len(paragraphs[0])

        for i in range(1, len(paragraphs)):
            sim = _cosine_sim(embeddings[i - 1], embeddings[i])
            added_len = len(paragraphs[i]) + 2
            fits = current_len + added_len <= self.chunk_size
            if sim >= self.similarity_threshold and fits:
                current.append(paragraphs[i])
                current_len += added_len
            else:
                groups.append("\n\n".join(current))
                current = [paragraphs[i]]
                current_len = len(paragraphs[i])

        if current:
            groups.append("\n\n".join(current))

        return groups


# ---------------------------------------------------------------------------
# Приватные вспомогательные функции
# ---------------------------------------------------------------------------

def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Разбить текст на пары (заголовок, тело секции) по Markdown-заголовкам."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append(("", preamble))

    for idx, match in enumerate(matches):
        header = match.group(0)
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((header, body))

    return sections


def _make_id(text: str, index: int) -> str:
    """Сгенерировать стабильный id чанка из содержимого и позиции."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"chunk_{index}_{h}"


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
