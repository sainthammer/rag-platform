"""Конкретные реализации DocumentLoader для разных форматов файлов.

Каждый загрузчик извлекает текст из своего формата и возвращает
чистую строку для последующей передачи в Chunker.
"""

from __future__ import annotations

import re
from pathlib import Path

from .ports import DocumentLoader


class TextLoader(DocumentLoader):
    """Загружает обычные текстовые файлы.

    Args:
        encoding: Кодировка файла. По умолчанию UTF-8.
        errors: Политика обработки ошибок кодировки (``replace``, ``ignore``, ``strict``).
    """

    def __init__(self, encoding: str = "utf-8", errors: str = "replace") -> None:
        self.encoding = encoding
        self.errors = errors

    def load(self, path: str | Path) -> str:
        """Прочитать текстовый файл и вернуть содержимое как строку.

        Args:
            path: Путь к текстовому файлу.

        Returns:
            Содержимое файла.
        """
        return Path(path).read_text(encoding=self.encoding, errors=self.errors)


class MarkdownLoader(DocumentLoader):
    """Загружает Markdown-файлы, опционально удаляя разметку.

    Args:
        strip_markup: Если ``True`` — убирает заголовки, жирный, курсив,
            ссылки, изображения, кодовые блоки и оставляет чистый текст.
            По умолчанию ``False`` — разметка сохраняется (полезно для ByHeaderChunker).
    """

    def __init__(self, strip_markup: bool = False) -> None:
        self.strip_markup = strip_markup

    def load(self, path: str | Path) -> str:
        """Загрузить Markdown-файл.

        Args:
            path: Путь к .md-файлу.

        Returns:
            Текст файла, опционально очищенный от Markdown-разметки.
        """
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        if not self.strip_markup:
            return text

        # Удаляем кодовые блоки (``` ... ```)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        # Удаляем инлайн-код (`...`)
        text = re.sub(r"`[^`]+`", "", text)
        # Удаляем изображения ![alt](url)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # Ссылки [text](url) → text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Жирный и курсив ***...*** / **...** / *...*
        text = re.sub(r"[*_]{1,3}([^*_\n]+)[*_]{1,3}", r"\1", text)
        # Заголовки # ... → пустая строка (убираем маркер, текст остаётся)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Убираем горизонтальные линии
        text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        return text


class HTMLLoader(DocumentLoader):
    """Загружает HTML-файлы, извлекая чистый текст через BeautifulSoup.

    Зависимость: ``beautifulsoup4``. Если не установлен — при вызове
    ``load()`` будет raised ``ImportError``.

    Args:
        parser: HTML-парсер для BeautifulSoup (``html.parser``, ``lxml``, ``html5lib``).
    """

    def __init__(self, parser: str = "html.parser") -> None:
        self.parser = parser

    def load(self, path: str | Path) -> str:
        """Загрузить HTML-файл и извлечь из него текст.

        Удаляет теги ``<script>``, ``<style>`` и ``<head>`` перед извлечением.

        Args:
            path: Путь к .html-файлу.

        Returns:
            Текстовое содержимое страницы без HTML-разметки.
        """
        from bs4 import BeautifulSoup  # noqa: PLC0415

        html = Path(path).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, self.parser)
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)


class PDFLoader(DocumentLoader):
    """Загружает PDF-файлы и извлекает текст через pypdf.

    Зависимость: ``pypdf``. Страницы объединяются через двойной перенос строки.
    """

    def load(self, path: str | Path) -> str:
        """Извлечь текст из PDF-документа постранично.

        Args:
            path: Путь к .pdf-файлу.

        Returns:
            Объединённый текст всех страниц, разделённый пустой строкой.
            Пустые страницы пропускаются.
        """
        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(page for page in pages if page.strip())
