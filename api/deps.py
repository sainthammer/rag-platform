"""FastAPI-зависимости для извлечения компонентов из app.state.

Все компоненты (pipeline, vector_db, llm) инициализируются один раз
в lifespan-функции app.py и хранятся в ``app.state``.
Зависимости здесь — тонкие функции-геттеры, которые роутеры получают
через ``Depends()``.
"""

from fastapi import Request

from embeddings.ports import EmbeddingService
from llm.ports import LLMProvider
from retrieval.pipeline import RAGPipeline
from vector_store.ports import VectorDB


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline  # type: ignore[no-any-return]


def get_vector_db(request: Request) -> VectorDB:
    return request.app.state.vector_db  # type: ignore[no-any-return]


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm  # type: ignore[no-any-return]


def get_embed_service(request: Request) -> EmbeddingService:
    return request.app.state.embed_service  # type: ignore[no-any-return]
