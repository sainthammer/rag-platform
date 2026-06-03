"""Централизованная конфигурация приложения через переменные окружения.

Все настройки читаются из окружения или файла ``.env`` (через pydantic-settings).
Единственный экземпляр ``settings`` создаётся при импорте модуля и используется
во всём приложении как глобальный объект-конфигурация.

Фабрики:
    ``build_llm_provider()``  — создаёт LLMProvider по LLM_PROVIDER.
    ``build_vector_db()``     — создаёт VectorDB по VECTOR_STORE_BACKEND.
    ``build_embed_fn()``      — создаёт функцию получения эмбеддингов.
"""

from typing import TYPE_CHECKING, Callable, Literal
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from llm.ports import LLMProvider
    from vector_store.ports import VectorDB


class Settings(BaseSettings):
    """Схема всех настроек приложения.

    pydantic-settings автоматически:
    - читает значения из переменных окружения (приоритет выше);
    - читает значения из файла ``.env`` (если файл существует);
    - валидирует типы и применяет ``default``, если переменная не задана.

    ``extra="ignore"`` означает: неизвестные переменные окружения молча
    игнорируются (не вызывают ошибки), что удобно в production-окружениях
    с большим числом глобальных переменных.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------
    # LLM — выбор провайдера и модели
    # ------------------------------------------------------------------

    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="", alias="LLM_MODEL")

    # ------------------------------------------------------------------
    # LLM — учётные данные и endpoints
    # ------------------------------------------------------------------

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    embedding_model: str = "text-embedding-3-small"
    # Должна совпадать с размерностью коллекций в векторном хранилище.
    embedding_dimension: int = 1536

    # ------------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------------

    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    # Имя коллекции по умолчанию, используемой в RAGPipeline.
    default_collection: str = Field(default="default", alias="DEFAULT_COLLECTION")
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    # Если задан — ChromaDB использует локальное файловое хранилище (PersistentClient)
    # вместо HttpClient. Удобно для разработки без Docker.
    chroma_persist_dir: str = Field(default="", alias="CHROMA_PERSIST_DIR")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_secret_key: str = "change-me-in-production"  # ОБЯЗАТЕЛЬНО менять в prod
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 9090


# ---------------------------------------------------------------------------
# Дефолты моделей
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.2",
}


# ---------------------------------------------------------------------------
# Фабрики компонентов
# ---------------------------------------------------------------------------

def build_llm_provider(s: "Settings | None" = None) -> "LLMProvider":
    """Создать LLMProvider по настройкам LLM_PROVIDER / LLM_MODEL.

    Args:
        s: Объект настроек. ``None`` → используется глобальный ``settings``.

    Returns:
        Готовый к использованию LLMProvider.
    """
    if s is None:
        s = settings

    model = s.llm_model or _PROVIDER_DEFAULTS[s.llm_provider]

    if s.llm_provider == "openai":
        from llm.adapters import OpenAIProvider
        return OpenAIProvider(model=model, api_key=s.openai_api_key or None)

    if s.llm_provider == "anthropic":
        from llm.adapters import AnthropicProvider
        return AnthropicProvider(model=model, api_key=s.anthropic_api_key or None)

    from llm.adapters import OllamaProvider
    return OllamaProvider(model=model, base_url=s.ollama_base_url)


def build_vector_db(s: "Settings | None" = None, collection: str | None = None) -> "VectorDB":
    """Создать VectorDB по настройкам VECTOR_STORE_BACKEND.

    Args:
        s: Объект настроек. ``None`` → используется глобальный ``settings``.
        collection: Имя коллекции. ``None`` → берётся ``settings.default_collection``.

    Returns:
        Готовый к использованию VectorDB.
    """
    if s is None:
        s = settings

    col = collection or s.default_collection

    if s.vector_store_backend == "qdrant":
        from vector_store.adapters import QdrantDB

        parsed = urlparse(s.qdrant_url)
        return QdrantDB(
            collection=col,
            vector_size=s.embedding_dimension,
            host=parsed.hostname or "localhost",
            port=parsed.port or 6333,
        )

    from vector_store.adapters import ChromaDB
    return ChromaDB(
        collection=col,
        persist_directory=s.chroma_persist_dir or None,
        host=s.chroma_host,
        port=s.chroma_port,
    )


def build_embed_fn(s: "Settings | None" = None) -> Callable[[str], list[float]]:
    """Создать функцию получения эмбеддинга для произвольного текста.

    Если задан OPENAI_API_KEY — использует OpenAI Embeddings API.
    Иначе возвращает нулевой вектор нужной размерности (для тестов / Ollama).

    Args:
        s: Объект настроек. ``None`` → используется глобальный ``settings``.

    Returns:
        Callable ``(text: str) -> list[float]``.
    """
    if s is None:
        s = settings

    if s.openai_api_key:
        from openai import OpenAI

        client = OpenAI(api_key=s.openai_api_key)
        model = s.embedding_model
        dim = s.embedding_dimension

        def _openai_embed(text: str) -> list[float]:
            resp = client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding

        return _openai_embed

    # Нулевой вектор — заглушка для локальной разработки без API-ключа.
    dim = s.embedding_dimension

    def _zero_embed(text: str) -> list[float]:
        return [0.0] * dim

    return _zero_embed


# Глобальный синглтон — создаётся один раз при импорте модуля.
settings = Settings()
