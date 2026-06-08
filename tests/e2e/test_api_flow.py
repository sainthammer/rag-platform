"""E2E-тест: ingest → search → ask.

Поднимает полное FastAPI-приложение с:
  - FakeEmbeddingService (детерминированные векторы, без ML-модели)
  - MockLLM (возвращает предсказуемый ответ)
  - ChromaDB в tmp_path (без постоянного хранилища)

Проверяет:
  1. POST /v1/ingest — документ принят (202), фоновая задача завершается.
  2. GET /v1/ingest/{job_id} — статус «done», chunks_indexed > 0.
  3. POST /v1/search — запрос по слову из документа возвращает результаты.
  4. POST /v1/ask — ответ содержит непустой список sources.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from embeddings.adapters import FakeEmbeddingService
from llm.llm_dataclasses import Message
from llm.ports import LLMProvider
from vector_store.adapters import ChromaDB


# ---------------------------------------------------------------------------
# Фиктивные компоненты
# ---------------------------------------------------------------------------

class _MockLLM(LLMProvider):
    async def complete(self, messages: list[Message], stream: bool = False) -> str:
        return "Ответ от MockLLM"


_FAKE_EMBED = FakeEmbeddingService(size=8, normalize=True)

_DOC_TEXT = (
    "Python — высокоуровневый язык программирования с динамической типизацией. "
    "Он поддерживает объектно-ориентированное, функциональное и процедурное программирование. "
    "Python широко используется в анализе данных, машинном обучении и веб-разработке."
)


# ---------------------------------------------------------------------------
# Фикстура приложения
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path: Path, monkeypatch):
    """FastAPI-приложение с заглушками вместо реальных сервисов.

    ASGITransport не запускает ASGI lifespan, поэтому app.state заполняем
    вручную. api/__init__.py затеняет субмодуль — получаем его через sys.modules.
    """
    import sys
    import api.app  # гарантируем, что модуль загружен
    app_mod = sys.modules["api.app"]

    chroma_dir = str(tmp_path / "chroma")

    def _fake_vector_db(settings_or_collection=None, collection=None):
        # Called as build_vector_db(settings, collection=name) from routers,
        # OR as vector_db_factory(collection) (positional) from RAGPipeline._retrieve.
        col = collection or (
            settings_or_collection if isinstance(settings_or_collection, str) else None
        ) or "default"
        return ChromaDB(col, persist_directory=chroma_dir)

    # Патчим build_vector_db в роутерах, которые вызывают его напрямую
    import api.routers.ingest as ingest_mod
    import api.routers.search as search_mod
    monkeypatch.setattr(ingest_mod, "build_vector_db", _fake_vector_db)
    monkeypatch.setattr(search_mod, "build_vector_db", _fake_vector_db)

    # Строим фейковый пайплайн для /ask
    from retrieval.pipeline import RAGPipeline
    _mock_llm = _MockLLM()
    _fake_pipeline = RAGPipeline(
        llm=_mock_llm,
        vector_db_factory=_fake_vector_db,
        embed_fn=_FAKE_EMBED.embed,
    )

    fastapi_app = app_mod.app

    # ASGITransport не запускает lifespan — заполняем state вручную
    fastapi_app.state.embed_service = _FAKE_EMBED
    fastapi_app.state.llm = _mock_llm
    fastapi_app.state.vector_db = _fake_vector_db()
    fastapi_app.state.store_client = None
    fastapi_app.state.store_backend = "chroma"
    fastapi_app.state.pipeline = _fake_pipeline

    # Отключаем аутентификацию через dependency_overrides
    from api.middleware.auth import require_auth
    fastapi_app.dependency_overrides[require_auth] = lambda: None

    yield fastapi_app

    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture()
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# E2E-тест
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_search_ask_flow(client: AsyncClient) -> None:
    """Полный цикл: загрузить документ → найти → спросить → проверить sources."""

    # 1. Ingest — отправляем текст как form-поле
    resp = await client.post(
        "/v1/ingest",
        data={
            "text": _DOC_TEXT,
            "collection": "e2e_test",
            "doc_id": "python_doc",
            "chunk_size": "150",
            "chunk_overlap": "20",
        },
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    # 2. Ждём завершения фоновой задачи (до 3 секунд)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        status_resp = await client.get(f"/v1/ingest/{job_id}")
        assert status_resp.status_code == 200
        job = status_resp.json()
        if job["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.05)

    assert job["status"] == "done", f"Ingest завершился с ошибкой: {job.get('error')}"
    assert job["chunks_indexed"] > 0, "Нет проиндексированных чанков"

    # 3. Search — ищем слово из документа
    resp = await client.post(
        "/v1/search",
        json={"query": "Python язык программирования", "collection": "e2e_test", "n_results": 3},
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) > 0, "Поиск вернул 0 результатов"
    assert all("text" in r and "score" in r for r in results)

    # 4. Ask — задаём вопрос
    resp = await client.post(
        "/v1/ask",
        json={"question": "Что такое Python?", "collection": "e2e_test", "stream": False},
    )
    assert resp.status_code == 200, resp.text
    ask_body = resp.json()
    assert ask_body["answer"], "Ответ пуст"
    assert isinstance(ask_body["sources"], list), "sources должен быть списком"
    assert len(ask_body["sources"]) > 0, "sources пуст — retrieval не вернул чанки"
    assert ask_body["confidence"] > 0.0


@pytest.mark.asyncio
async def test_ask_stream_returns_sse(client: AsyncClient) -> None:
    """POST /v1/ask с stream=true возвращает text/event-stream."""
    resp = await client.post(
        "/v1/ask",
        json={"question": "Что такое Python?", "collection": "e2e_test", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_ingest_job_not_found(client: AsyncClient) -> None:
    """GET /v1/ingest/{unknown} возвращает 404."""
    resp = await client.get("/v1/ingest/nonexistent-job-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ingest_no_source_returns_422(client: AsyncClient) -> None:
    """POST /v1/ingest без источника возвращает 422."""
    resp = await client.post("/v1/ingest", data={"collection": "test"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_eval_run_returns_job_id(client: AsyncClient) -> None:
    """POST /v1/eval/run возвращает job_id и status=pending."""
    resp = await client.post(
        "/v1/eval/run",
        json={"mode": "mock", "max_cases": 2},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "pending"
