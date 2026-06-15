# Модуль 2 — chunking/

**Файлы:** `chunking/ports.py`, `chunking/loaders.py`, `chunking/adapters.py`, `chunking/ingest.py`

Модуль отвечает за загрузку документов и разбивку их на части (чанки). Полная архитектура Ports & Adapters.

## Зачем резать на чанки

LLM и embedding-модели имеют ограничение на размер входных данных. Кроме того, векторный поиск работает точнее на небольших смысловых фрагментах — чем меньше чанк, тем точнее его вектор отражает конкретную идею.

## Структура

```
chunking/
  ports.py    ← абстракции: Chunk, Chunker, DocumentLoader
  loaders.py  ← загрузчики: Text, Markdown, HTML, PDF
  adapters.py ← стратегии разбивки: Fixed, ByHeader, Semantic
  ingest.py   ← фасад: ingest(path, strategy, ...)
```

## ports.py — три абстракции

### `Chunk` — датакласс чанка

```python
@dataclass
class Chunk:
    text: str                    # текст фрагмента
    metadata: dict[str, Any]     # метаданные (источник, индекс, заголовок...)
    id: str                      # стабильный id: sha256(text)[:12] + индекс
```

### `Chunker` — интерфейс стратегии разбивки

```python
class Chunker(ABC):
    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]: ...
```

### `DocumentLoader` — интерфейс загрузчика формата

```python
class DocumentLoader(ABC):
    def load(self, path: str | Path) -> str: ...
```

## loaders.py — загрузчики по форматам

| Класс | Форматы | Особенности |
|---|---|---|
| `TextLoader` | `.txt` | UTF-8, настраиваемая политика ошибок кодировки |
| `MarkdownLoader` | `.md`, `.markdown` | `strip_markup=True` — убирает `#`, `**`, `[](url)`, кодовые блоки |
| `HTMLLoader` | `.html`, `.htm` | BeautifulSoup: удаляет `<script>`, `<style>`, `<head>` |
| `PDFLoader` | `.pdf` | pypdf: постраничное извлечение, пустые страницы пропускаются |

Пример `MarkdownLoader` с `strip_markup=True`:
```
# Заголовок                →  ""  (маркер удалён, текст остался в следующей строке)
**жирный** и *курсив*      →  "жирный и курсив"
[текст](https://url.com)   →  "текст"
```Code```                  →  ""
```

## adapters.py — три стратегии разбивки

### `FixedSizeChunker` — скользящее окно

Классический подход. Использует `RecursiveCharacterTextSplitter` из langchain:

```
Приоритет разбивки: \n\n → \n → ". " → " " → символы
```

```python
chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
```

**Перекрытие (overlap)** — важный параметр. Если смысл "на стыке" двух чанков, перекрытие гарантирует что контекст не теряется:

```
[  чанк 1  ][  чанк 2  ][  чанк 3  ]
         [░░]        [░░]             ← 50 символов перекрытия
```

Подходит для любого текста.

### `ByHeaderChunker` — по заголовкам Markdown

Каждый заголовок `#`, `##`, `###` — новая секция. Текст до первого заголовка (преамбула) тоже становится чанком.

```python
chunker = ByHeaderChunker(chunk_size=1000, min_chunk_size=30)
```

Если секция длиннее `chunk_size` — она дополнительно дробится `FixedSizeChunker`'ом.

Метаданные каждого чанка содержат `"header"` — текст заголовка, под которым находится фрагмент. Это полезно для поиска: можно показать пользователю из какого раздела документа взят ответ.

Подходит для структурированных Markdown-документов (README, документация, статьи).

### `SemanticChunker` — по смысловой близости

Объединяет параграфы (разделённые `\n\n`) в чанки на основе семантической близости.

```python
chunker = SemanticChunker(chunk_size=500, similarity_threshold=0.85, embed_fn=svc.embed)
```

**Без `embed_fn`** — жадное объединение по размеру: параграфы объединяются пока не превысят `chunk_size`.

**С `embed_fn`** — соседние параграфы объединяются если `cosine_similarity >= threshold`:

```
параграф_1 ──embed──▶ vec_1 ─┐
параграф_2 ──embed──▶ vec_2 ─┴─ cosine_sim ≥ 0.85? → объединить
параграф_3 ──embed──▶ vec_3 ─┐
параграф_4 ──embed──▶ vec_4 ─┴─ cosine_sim < 0.85? → новый чанк
```

Подходит для текстов с плавными переходами между темами (научные статьи, эссе).

## ingest.py — фасад пайплайна

Функция `ingest()` связывает лоадер и чанкер:

```python
chunks = ingest(
    source_path="docs/manual.pdf",
    strategy="by_header",   # fixed / by_header / semantic
    chunk_size=1000,
    chunk_overlap=50,
    metadata={"project": "my-project"},
)
```

Внутри:
1. Определяет лоадер по расширению файла
2. `loader.load(path)` → текст
3. Создаёт нужный `Chunker` по `strategy`
4. `chunker.split(text, base_meta)` → список `Chunk`

Метаданные `source` (путь к файлу) и `filename` добавляются автоматически.

**Маппинг расширений → лоадер:**
```python
{
    ".txt":      TextLoader,
    ".md":       MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".html":     HTMLLoader,
    ".htm":      HTMLLoader,
    ".pdf":      PDFLoader,
}
```

## Где используется

В `api/routers/ingest.py` — при загрузке через API (пока использует старую `chunk_text()` из `chunking/__init__.py`).

В `mcp/rag_server.py` — инструмент `ingest_document`.

В `chunking/ingest.py` — напрямую при работе со скриптами (`compare_chunks.py`, `reindex.py`).

## Когда какую стратегию выбирать

| Стратегия | Когда |
|---|---|
| `fixed` | Любой текст, универсально, быстро |
| `by_header` | Markdown-документация, README, статьи с заголовками |
| `semantic` | Непрерывный текст без явной структуры, требует embedding |
