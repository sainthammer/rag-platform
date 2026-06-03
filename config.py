"""Централизованная конфигурация приложения через переменные окружения.

Все настройки читаются из окружения или файла ``.env`` (через pydantic-settings).
Единственный экземпляр ``settings`` создаётся при импорте модуля и используется
во всём приложении как глобальный объект-конфигурация.

Выбор провайдера LLM:
    Устанавливается переменной ``LLM_PROVIDER=openai|anthropic|ollama``.
    Фабрика ``build_llm_provider()`` читает эту переменную и возвращает
    готовый экземпляр нужного адаптера.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# TYPE_CHECKING позволяет использовать LLMProvider в аннотации без импорта
# в рантайме — это разрывает циклический импорт (config → llm → config).
if TYPE_CHECKING:
    from llm.ports import LLMProvider


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

    # Какой провайдер использовать. Значение ограничено Literal, чтобы
    # pydantic выдал ошибку при опечатке в .env вместо молчаливого неверного поведения.
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    # Имя конкретной модели. Пустая строка = использовать дефолт из _PROVIDER_DEFAULTS.
    llm_model: str = Field(default="", alias="LLM_MODEL")

    # ------------------------------------------------------------------
    # LLM — учётные данные и endpoints
    # ------------------------------------------------------------------

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Ollama работает локально и не требует ключа — только URL сервера.
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # должна совпадать с размером коллекции в векторном хранилище

    # ------------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------------

    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""  # пустая строка = публичный/локальный Qdrant без аутентификации

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_secret_key: str = "change-me-in-production"  # ОБЯЗАТЕЛЬНО менять в prod

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 9090


# Дефолтные модели для каждого провайдера.
# Используются в build_llm_provider(), когда LLM_MODEL не задан в .env.
_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.2",
}


def build_llm_provider(s: Settings | None = None) -> "LLMProvider":
    """Создать экземпляр LLMProvider на основе текущей конфигурации.

    Фабричная функция читает ``settings.llm_provider`` и возвращает готовый
    адаптер. Импорты адаптеров выполняются лениво внутри веток ``if``, чтобы
    не тянуть все SDK при любом запуске.

    Args:
        s: Объект настроек. Если ``None`` — используется глобальный ``settings``.
            Явная передача полезна в тестах, где нужно проверить поведение
            с конкретными значениями без изменения env.

    Returns:
        Готовый к использованию экземпляр ``LLMProvider``.

    Example::

        # В production-коде:
        llm = build_llm_provider()

        # В тестах:
        llm = build_llm_provider(Settings(llm_provider="ollama", llm_model="mistral"))
    """
    if s is None:
        s = settings  # берём глобальный синглтон, если объект не передан

    # Если LLM_MODEL не задан — подставляем разумный дефолт для провайдера.
    model = s.llm_model or _PROVIDER_DEFAULTS[s.llm_provider]

    if s.llm_provider == "openai":
        from llm.adapters import OpenAIProvider

        # ``or None``: pydantic хранит пустую строку как "", но SDK ожидает None
        # при отсутствии ключа (чтобы прочитать OPENAI_API_KEY из env самостоятельно).
        return OpenAIProvider(model=model, api_key=s.openai_api_key or None)

    if s.llm_provider == "anthropic":
        from llm.adapters import AnthropicProvider

        return AnthropicProvider(model=model, api_key=s.anthropic_api_key or None)

    # Единственный оставшийся вариант — "ollama".
    from llm.adapters import OllamaProvider

    return OllamaProvider(model=model, base_url=s.ollama_base_url)


# Глобальный синглтон конфигурации — создаётся один раз при импорте модуля.
# Весь код приложения должен использовать именно его, а не создавать Settings() заново.
settings = Settings()
