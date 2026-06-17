"""Стратегии чанкинга для загруженных документов."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .dataclasses import Chunk
from .ports import Chunker
from .utils import normalize_text, sliding_windows, stable_hash

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class FixedSizeChunker(Chunker):
    """Разбивает текст на символьные окна фиксированного размера."""

    chunk_size: int = 1000
    overlap: int = 150

    def chunk(self, text: str, metadata: dict[str, object] | None = None) -> list[Chunk]:
        """Разбить текст на fixed-size чанки.

        Args:
            text: Исходный текст документа.
            metadata: Метаданные источника, которые будут добавлены в каждый чанк.

        Returns:
            Список чанков с полями ``chunk_index``, ``chunk_start``,
            ``chunk_end`` и ``chunk_strategy`` в metadata.

        Raises:
            ValueError: Если ``chunk_size`` некорректен или ``overlap`` отрицательный.
        """
        base_metadata = dict(metadata or {})
        normalized = normalize_text(text)
        chunks: list[Chunk] = []
        overlap = _effective_overlap(self.chunk_size, self.overlap)
        windows = sliding_windows(normalized, self.chunk_size, overlap)
        for index, (start, piece) in enumerate(windows):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": index,
                "chunk_start": start,
                "chunk_end": start + len(piece),
                "chunk_strategy": "fixed",
            }
            chunks.append(_make_chunk(piece, chunk_metadata))
        return chunks


@dataclass
class ByHeaderChunker(Chunker):
    """Делит Markdown-like текст по заголовкам и режет крупные секции."""

    chunk_size: int = 1000
    overlap: int = 150

    def chunk(self, text: str, metadata: dict[str, object] | None = None) -> list[Chunk]:
        """Разбить текст по Markdown-заголовкам.

        Args:
            text: Исходный Markdown-like текст.
            metadata: Метаданные источника, которые будут добавлены в каждый чанк.

        Returns:
            Список чанков с метаданными секции: ``section``, ``section_index``,
            ``chunk_index`` и ``chunk_strategy``.

        Raises:
            ValueError: Если ``chunk_size`` некорректен или ``overlap`` отрицательный.
        """
        base_metadata = dict(metadata or {})
        sections = self._sections(normalize_text(text))
        chunks: list[Chunk] = []

        for section_index, (header, body) in enumerate(sections):
            section_metadata = {
                **base_metadata,
                "section": header,
                "section_index": section_index,
                "chunk_strategy": "by_header",
            }
            overlap = _effective_overlap(self.chunk_size, self.overlap)
            for _, piece in sliding_windows(body, self.chunk_size, overlap):
                chunk_metadata = {**section_metadata, "chunk_index": len(chunks)}
                chunks.append(_make_chunk(piece, chunk_metadata))
        return chunks

    def _sections(self, text: str) -> list[tuple[str | None, str]]:
        """Выделить секции по Markdown-заголовкам.

        Args:
            text: Нормализованный Markdown-like текст.

        Returns:
            Список пар ``(header, body)``, где ``header`` может быть ``None``.
        """
        sections: list[tuple[str | None, list[str]]] = []
        current_header: str | None = None
        current_lines: list[str] = []

        for line in text.splitlines():
            header_match = HEADER_RE.match(line)
            if header_match and current_lines:
                sections.append((current_header, current_lines))
                current_lines = []
            if header_match:
                current_header = header_match.group(2).strip()
            current_lines.append(line)

        if current_lines:
            sections.append((current_header, current_lines))

        result: list[tuple[str | None, str]] = []
        for header, lines in sections:
            body = "\n".join(lines).strip()
            if body:
                result.append((header, body))
        return result


@dataclass
class SemanticChunker(Chunker):
    """Делит текст по абзацам и предложениям с учетом лимита размера."""

    chunk_size: int = 1000
    overlap_sentences: int = 1

    def chunk(self, text: str, metadata: dict[str, object] | None = None) -> list[Chunk]:
        """Разбить текст на семантические чанки.

        Args:
            text: Исходный текст документа.
            metadata: Метаданные источника, которые будут добавлены в каждый чанк.

        Returns:
            Список чанков, собранных из абзацев или предложений без превышения
            ``chunk_size``, насколько это возможно.

        Raises:
            ValueError: Если ``chunk_size`` меньше или равен нулю либо
                ``overlap_sentences`` отрицательный.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size должен быть больше 0")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences не может быть отрицательным")

        base_metadata = dict(metadata or {})
        units = self._semantic_units(normalize_text(text))
        chunks: list[Chunk] = []
        current: list[str] = []
        current_length = 0

        for unit in units:
            separator = "\n\n" if "\n" in unit else " "
            projected = current_length + len(unit) + (len(separator) if current else 0)
            if current and projected > self.chunk_size:
                self._append_chunk(chunks, current, base_metadata)
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current_length = sum(len(item) for item in current)

            if len(unit) > self.chunk_size:
                for _, piece in sliding_windows(unit, self.chunk_size, 0):
                    self._append_chunk(chunks, [piece], base_metadata)
                current = []
                current_length = 0
                continue

            current.append(unit)
            current_length += len(unit)

        if current:
            self._append_chunk(chunks, current, base_metadata)
        return chunks

    def _semantic_units(self, text: str) -> list[str]:
        """Получить смысловые единицы для сборки чанков.

        Args:
            text: Нормализованный текст документа.

        Returns:
            Список абзацев или предложений. Крупные абзацы дополнительно
            разбиваются по простым границам предложений.
        """
        paragraphs = [
            paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()
        ]
        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                units.append(paragraph)
                continue
            units.extend(
                sentence.strip()
                for sentence in SENTENCE_RE.split(paragraph)
                if sentence.strip()
            )
        return units

    def _append_chunk(
        self,
        chunks: list[Chunk],
        units: list[str],
        base_metadata: dict[str, object],
    ) -> None:
        """Добавить чанк, собранный из смысловых единиц.

        Args:
            chunks: Список, в который нужно добавить новый чанк.
            units: Абзацы или предложения, составляющие текст чанка.
            base_metadata: Метаданные источника.

        Returns:
            ``None``. Список ``chunks`` изменяется на месте.
        """
        text = "\n\n".join(units).strip()
        if not text:
            return
        chunk_metadata = {
            **base_metadata,
            "chunk_index": len(chunks),
            "chunk_strategy": "semantic",
        }
        chunks.append(_make_chunk(text, chunk_metadata))


def build_chunker(strategy: str | Chunker = "fixed", chunk_size: int = 1000) -> Chunker:
    """Создать chunker по названию стратегии или вернуть готовый объект.

    Args:
        strategy: Название стратегии: ``fixed``, ``fixed_size``, ``header``,
            ``by_header``, ``semantic`` или ``semantic_chunker``. Можно передать
            уже созданный объект ``Chunker``.
        chunk_size: Целевой максимальный размер чанка в символах.

    Returns:
        Экземпляр ``Chunker``.

    Raises:
        ValueError: Если строковое название стратегии неизвестно.
    """
    if isinstance(strategy, Chunker):
        return strategy

    normalized = strategy.lower().replace("-", "_")
    if normalized in {"fixed", "fixed_size"}:
        return FixedSizeChunker(chunk_size=chunk_size)
    if normalized in {"header", "by_header"}:
        return ByHeaderChunker(chunk_size=chunk_size)
    if normalized in {"semantic", "semantic_chunker"}:
        return SemanticChunker(chunk_size=chunk_size)
    raise ValueError(f"Неизвестная стратегия чанкинга: {strategy!r}")


def _make_chunk(text: str, metadata: dict[str, object]) -> Chunk:
    """Создать чанк со стабильным id.

    Args:
        text: Текст чанка.
        metadata: Метаданные чанка.

    Returns:
        Объект ``Chunk`` со стабильным hash-based id.
    """
    chunk_id = stable_hash(metadata.get("source", ""), metadata.get("chunk_index", 0), text)
    return Chunk(text=text, metadata=metadata, id=chunk_id)


def _effective_overlap(chunk_size: int, overlap: int) -> int:
    """Нормализовать значение overlap для fixed-window стратегий.

    Args:
        chunk_size: Максимальный размер чанка в символах.
        overlap: Запрошенный overlap между соседними чанками.

    Returns:
        Безопасное значение overlap. Если overlap не меньше ``chunk_size``,
        используется 20% от размера чанка.

    Raises:
        ValueError: Если ``overlap`` отрицательный.
    """
    if overlap < 0:
        raise ValueError("overlap не может быть отрицательным")
    if overlap >= chunk_size:
        return max(0, chunk_size // 5)
    return overlap
