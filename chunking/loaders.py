"""Загрузчики документов для ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from .ports import DocumentLoader
from .utils import normalize_text, read_text_file, stable_hash


class TextLoader(DocumentLoader):
    """Загружает plain text файлы."""

    content_type = "text/plain"

    def load(self, source_path: str | Path) -> tuple[str, dict[str, object]]:
        """Загрузить текстовый файл.

        Args:
            source_path: Путь к ``.txt`` или другому текстовому файлу.

        Returns:
            Кортеж из нормализованного текста и метаданных файла.

        Raises:
            OSError: Если файл нельзя открыть или прочитать.
            UnicodeDecodeError: Если файл не декодируется как UTF-8.
        """
        path = Path(source_path)
        text = normalize_text(read_text_file(path))
        return text, _file_metadata(path, self.content_type, text)


class MarkdownLoader(TextLoader):
    """Загружает Markdown как plain text с сохранением заголовков."""

    content_type = "text/markdown"


class HTMLLoader(DocumentLoader):
    """Загружает HTML и извлекает видимый текст страницы."""

    def load(self, source_path: str | Path) -> tuple[str, dict[str, object]]:
        """Загрузить HTML-файл и удалить служебные теги.

        Args:
            source_path: Путь к ``.html`` или ``.htm`` файлу.

        Returns:
            Кортеж из видимого текста страницы и метаданных файла.

        Raises:
            ImportError: Если не установлен пакет ``beautifulsoup4``.
            OSError: Если файл нельзя открыть или прочитать.
            UnicodeDecodeError: Если файл не декодируется как UTF-8.
        """
        from bs4 import BeautifulSoup

        path = Path(source_path)
        html = read_text_file(path)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = normalize_text(soup.get_text(separator="\n"))
        return text, _file_metadata(path, "text/html", text)


class PDFLoader(DocumentLoader):
    """Загружает PDF через ``pdfminer.six``."""

    def load(self, source_path: str | Path) -> tuple[str, dict[str, object]]:
        """Извлечь текст из PDF-файла.

        Args:
            source_path: Путь к ``.pdf`` файлу.

        Returns:
            Кортеж из нормализованного текста PDF и метаданных файла.

        Raises:
            ImportError: Если не установлен пакет ``pdfminer.six``.
            OSError: Если файл нельзя открыть или прочитать.
        """
        from pdfminer.high_level import extract_text

        path = Path(source_path)
        text = normalize_text(extract_text(str(path)))
        return text, _file_metadata(path, "application/pdf", text)


def get_loader(source_path: str | Path) -> DocumentLoader:
    """Выбрать загрузчик по расширению файла.

    Args:
        source_path: Путь к документу.

    Returns:
        Экземпляр загрузчика для поддерживаемого типа файла.

    Raises:
        ValueError: Если расширение документа не поддерживается.
    """
    suffix = Path(source_path).suffix.lower()
    if suffix == ".pdf":
        return PDFLoader()
    if suffix in {".md", ".markdown"}:
        return MarkdownLoader()
    if suffix in {".html", ".htm"}:
        return HTMLLoader()
    if suffix in {".txt", ".text", ""}:
        return TextLoader()
    raise ValueError(f"Неподдерживаемый тип документа: {suffix or '<без расширения>'}")


def _file_metadata(path: Path, content_type: str, text: str) -> dict[str, object]:
    """Собрать метаданные исходного файла.

    Args:
        path: Путь к файлу.
        content_type: MIME-подобный тип содержимого.
        text: Нормализованный текст документа.

    Returns:
        Словарь с путем, именем файла, размером, временем изменения и hash.

    Raises:
        OSError: Если невозможно получить информацию о файле.
    """
    stat = path.stat()
    return {
        "source": str(path),
        "source_name": path.name,
        "content_type": content_type,
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "document_hash": stable_hash(text, length=64),
    }
