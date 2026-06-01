from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Vector Store
    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_secret_key: str = "change-me-in-production"

    # Observability
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 9090


settings = Settings()
