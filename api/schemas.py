"""Pydantic-схемы для всех запросов и ответов API.

Каждая схема содержит:
- ``description`` — назначение поля (видно в Swagger UI).
- ``examples``    — один конкретный пример значения (Pydantic v2).
- ``model_config`` с ``json_schema_extra`` — пример целого объекта для документации.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    status: Literal["ok", "error", "not_configured"] = Field(
        description="Статус компонента",
        examples=["ok"],
    )
    detail: str | None = Field(
        default=None,
        description="Дополнительная информация (версия, параметры подключения)",
        examples=["backend=chroma, vectors=1024"],
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "components": {
                    "vector_store": {"status": "ok", "detail": "backend=chroma, vectors=342"},
                    "llm": {"status": "ok", "detail": "OllamaProvider, url=http://localhost:11434/v1"},
                    "cache": {"status": "not_configured", "detail": None},
                },
            }
        }
    )

    status: Literal["ok", "degraded"] = Field(
        description="Общий статус: `ok` — все компоненты работают, `degraded` — есть проблемы",
    )
    components: dict[str, ComponentHealth] = Field(
        description="Статус каждого компонента (vector_store, llm, cache)",
    )


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class CollectionItem(BaseModel):
    name: str = Field(description="Имя коллекции", examples=["default"])
    vectors_count: int = Field(ge=0, description="Число векторов в коллекции", examples=[342])


class CollectionsResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collections": [
                    {"name": "default", "vectors_count": 342},
                    {"name": "bench_256", "vectors_count": 189},
                ]
            }
        }
    )
    collections: list[CollectionItem] = Field(description="Список коллекций")


class DeleteCollectionResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"deleted": "bench_256"}})
    deleted: str = Field(description="Имя удалённой коллекции", examples=["bench_256"])


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestJobResponse(BaseModel):
    """Ответ POST /v1/ingest — задача поставлена в очередь."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"job_id": "a3f1c2d4", "status": "pending"}}
    )

    job_id: str = Field(description="Уникальный идентификатор задачи индексации", examples=["a3f1c2d4"])
    status: Literal["pending", "running", "done", "error"] = Field(
        description="Текущий статус задачи", examples=["pending"]
    )


class IngestStatusResponse(BaseModel):
    """Ответ GET /v1/ingest/{job_id}."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "a3f1c2d4",
                "status": "done",
                "chunks_indexed": 12,
                "collection": "default",
                "error": None,
            }
        }
    )

    job_id: str = Field(description="Идентификатор задачи", examples=["a3f1c2d4"])
    status: Literal["pending", "running", "done", "error"] = Field(
        description="Статус задачи", examples=["done"]
    )
    chunks_indexed: int = Field(
        default=0,
        description="Число проиндексированных чанков (заполняется после завершения)",
        examples=[12],
    )
    collection: str = Field(
        default="",
        description="Коллекция, в которую записаны чанки",
        examples=["default"],
    )
    error: str | None = Field(
        default=None,
        description="Текст ошибки (только при status=error)",
        examples=[None],
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Что такое генератор в Python?",
                "collection": "default",
                "n_results": 5,
                "score_threshold": 0.0,
            }
        }
    )

    query: str = Field(
        min_length=1,
        description="Текстовый запрос для семантического поиска",
        examples=["Что такое генератор в Python?"],
    )
    collection: str = Field(
        default="default",
        description="Имя коллекции в векторном хранилище",
        examples=["default"],
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Максимальное число возвращаемых результатов",
        examples=[5],
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Минимальный score (0–1). Результаты ниже порога отфильтровываются",
        examples=[0.0],
    )


class SearchResultItem(BaseModel):
    id: str = Field(description="Идентификатор чанка", examples=["docs/python.md_chunk3"])
    text: str = Field(description="Текст чанка", examples=["Генератор — это функция с yield..."])
    score: float = Field(description="Оценка релевантности [0, 1]", examples=[0.87])
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Метаданные чанка (источник, индекс и т.д.)",
        examples=[{"source": "docs/python.md", "chunk_index": 3}],
    )


class SearchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "id": "docs/python.md_chunk3",
                        "text": "Генератор — это функция с оператором yield...",
                        "score": 0.87,
                        "metadata": {"source": "docs/python.md", "chunk_index": 3},
                    }
                ],
                "query": "Что такое генератор в Python?",
                "collection": "default",
            }
        }
    )

    results: list[SearchResultItem] = Field(description="Найденные чанки, отсортированные по score убыванию")
    query: str = Field(description="Исходный запрос (echo)", examples=["Что такое генератор в Python?"])
    collection: str = Field(description="Коллекция, по которой выполнялся поиск", examples=["default"])


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Что такое декоратор в Python?",
                "collection": "default",
                "stream": False,
                "score_threshold": None,
                "n_results": 5,
            }
        }
    )

    question: str = Field(
        min_length=1,
        description="Вопрос на естественном языке",
        examples=["Что такое декоратор в Python?"],
    )
    collection: str = Field(
        default="default",
        description="Коллекция для retrieval",
        examples=["default"],
    )
    stream: bool = Field(
        default=False,
        description=(
            "Режим ответа: `false` → JSON `AskResponse`; "
            "`true` → `text/event-stream` (SSE) с токенами по мере генерации"
        ),
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Минимальный score retrieval. `null` — использовать значение из конфигурации пайплайна"
        ),
        examples=[None],
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Число чанков для retrieval",
        examples=[5],
    )


class AskSourceItem(BaseModel):
    text: str = Field(description="Текст чанка-источника", examples=["Декоратор — это функция..."])
    score: float = Field(description="Оценка релевантности [0, 1]", examples=[0.91])
    doc_id: str = Field(default="", description="Идентификатор чанка", examples=["docs/decorators.md_chunk1"])
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Метаданные чанка",
        examples=[{"source": "docs/decorators.md"}],
    )


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Декоратор — это функция высшего порядка, которая принимает другую функцию...",
                "sources": [
                    {
                        "text": "Декоратор — это функция...",
                        "score": 0.91,
                        "doc_id": "docs/decorators.md_chunk1",
                        "metadata": {"source": "docs/decorators.md"},
                    }
                ],
                "confidence": 0.91,
                "latency_ms": 312.4,
            }
        }
    )

    answer: str = Field(description="Сгенерированный ответ модели")
    sources: list[AskSourceItem] = Field(description="Чанки-источники, использованные в промпте")
    confidence: float = Field(
        description="Максимальный score среди retrieval-источников [0, 1]",
        examples=[0.91],
    )
    latency_ms: float = Field(
        description="Полная задержка RAG-цикла в миллисекундах",
        examples=[312.4],
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"mode": "mock", "max_cases": 5, "output_path": "eval_report.html"}
        }
    )

    mode: Literal["mock", "ollama"] = Field(
        default="mock",
        description=(
            "`mock` — быстрый прогон без внешних сервисов (RAGAS-метрики не считаются); "
            "`ollama` — реальная оценка через Ollama (требует `ollama serve`)"
        ),
    )
    max_cases: int = Field(
        default=5,
        ge=1,
        le=46,
        description="Максимальное число тест-кейсов для оценки (из 45 доступных)",
        examples=[5],
    )
    output_path: str = Field(
        default="eval_report.html",
        description="Путь для сохранения HTML-отчёта",
        examples=["eval_report.html"],
    )


class EvalJobResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"job_id": "e9b2c1a7", "status": "pending"}}
    )
    job_id: str = Field(description="Идентификатор задачи оценки", examples=["e9b2c1a7"])
    status: Literal["pending", "running", "done", "error"] = Field(
        description="Статус задачи", examples=["pending"]
    )


class EvalStatusResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "e9b2c1a7",
                "status": "done",
                "report_path": "eval_report.html",
                "hallucination_pass": 9,
                "hallucination_total": 10,
                "ragas_avg": {
                    "faithfulness": 0.82,
                    "answer_relevancy": 0.75,
                    "context_precision": 0.68,
                    "context_recall": 0.71,
                },
                "error": None,
            }
        }
    )

    job_id: str = Field(description="Идентификатор задачи", examples=["e9b2c1a7"])
    status: Literal["pending", "running", "done", "error"] = Field(
        description="Статус задачи", examples=["done"]
    )
    report_path: str | None = Field(
        default=None,
        description="Путь к HTML-отчёту (после завершения)",
        examples=["eval_report.html"],
    )
    hallucination_pass: int | None = Field(
        default=None,
        description="Число negative-кейсов, где модель корректно отказалась отвечать",
        examples=[9],
    )
    hallucination_total: int | None = Field(
        default=None,
        description="Общее число negative-кейсов (проверка галлюцинаций)",
        examples=[10],
    )
    ragas_avg: dict[str, float | None] | None = Field(
        default=None,
        description="Средние RAGAS-метрики (faithfulness, answer_relevancy, context_precision, context_recall)",
    )
    error: str | None = Field(
        default=None,
        description="Текст ошибки (при status=error)",
        examples=[None],
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"detail": "Неверные учётные данные"}})
    detail: str = Field(description="Описание ошибки", examples=["Неверные учётные данные"])
