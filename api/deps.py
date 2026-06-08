"""FastAPI-зависимости для извлечения компонентов из app.state.

Все компоненты (pipeline, vector_db, llm) инициализируются один раз
в lifespan-функции app.py и хранятся в ``app.state``.
Зависимости здесь — тонкие функции-геттеры, которые роутеры получают
через ``Depends()``.
"""

from fastapi import Request

from llm.ports import LLMProvider
from retrieval.pipeline import RAGPipeline
from vector_store.ports import VectorDB


def get_pipeline(request: Request) -> RAGPipeline:
    """Вернуть инициализированный RAGPipeline из app.state."""
    return request.app.state.pipeline  # type: ignore[no-any-return]


def get_vector_db(request: Request) -> VectorDB:
    """Вернуть VectorDB (per-collection) из app.state."""
    return request.app.state.vector_db  # type: ignore[no-any-return]


def get_llm(request: Request) -> LLMProvider:
    """Вернуть LLMProvider из app.state."""
    return request.app.state.llm  # type: ignore[no-any-return]
