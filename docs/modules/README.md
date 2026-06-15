# RAG Platform — Документация по модулям

Справочник по архитектуре и устройству проекта. Каждый файл — один модуль из плана изучения.

## Содержание

| Модуль | Файл | Тема |
|---|---|---|
| 0 | [module_00_rag_architecture.md](module_00_rag_architecture.md) | Что такое RAG и архитектура проекта |
| 1 | [module_01_config.md](module_01_config.md) | config.py — централизованная конфигурация |
| 2 | [module_02_chunking.md](module_02_chunking.md) | chunking/ — лоадеры, стратегии разбивки, пайплайн |
| 3 | [module_03_embeddings.md](module_03_embeddings.md) | embeddings/ — векторизация, кэш, fake-провайдер |
| 4 | [module_04_vector_store.md](module_04_vector_store.md) | vector_store/ — Chroma, Qdrant, BM25, гибридный поиск |
| 5 | [module_05_retrieval.md](module_05_retrieval.md) | retrieval/ — Retriever, реранкеры, RAGPipeline |
| 6 | [module_06_llm.md](module_06_llm.md) | llm/ — LLM-провайдеры, промпты, токенный бюджет |
| 7 | [module_07_api.md](module_07_api.md) | api/ — FastAPI, роутеры, middleware, rate limiting |
| 8 | [module_08_evaluation.md](module_08_evaluation.md) | evaluation/ — RAGAS метрики, тест-кейсы, отчёт |
| 9 | [module_09_observability.md](module_09_observability.md) | observability/ — Prometheus, OpenTelemetry, Grafana |
| 10 | [module_10_mcp.md](module_10_mcp.md) | mcp/ — MCP-сервер, интеграция с Claude |
| 11 | [module_11_tests.md](module_11_tests.md) | tests/ — unit, integration, e2e тесты |
| 12 | [module_12_infrastructure.md](module_12_infrastructure.md) | Dockerfile, docker-compose |

## Полный путь запроса

```
── Загрузка документа ──────────────────────────────────────────
POST /v1/ingest (файл / URL / текст)
  → chunking.ingest()        выбор лоадера по расширению (.pdf, .md, .html...)
  → Chunker.split()          стратегия: fixed / by_header / semantic
  → embed_service.embed_batch()   текст → векторы
  → db.add()                 сохранить в Chroma / Qdrant

── Вопрос пользователя ─────────────────────────────────────────
POST /v1/ask {"question": "...", "collection": "default"}
  → require_auth()           X-API-Key или JWT Bearer
  → RequestIDMiddleware      генерация X-Request-ID
  → rate limiter             проверка лимита по IP
  → pipeline.ask()
      → embed_fn(question)       вопрос → вектор
      → db.search()              векторный поиск
      → (опц.) reranker.rerank() CrossEncoder или MMR
      → budget.fit_chunks()      обрезка по токенному бюджету
      → template.format_*()      сборка промпта
      → llm.complete()           вызов LLM (OpenAI / Anthropic / Ollama)
  → AskResponse(answer, sources, confidence, latency_ms)

── Мониторинг ──────────────────────────────────────────────────
Prometheus  →  GET /v1/metrics  (скрейп метрик)
Jaeger      ←  OTLP gRPC        (трейсы каждого запроса)
Grafana     ←  Prometheus       (дашборды)
```

## Ключевые принципы архитектуры

| Принцип | Где видно |
|---|---|
| **Ports & Adapters** | `EmbeddingService`, `VectorDB`, `LLMProvider`, `Chunker`, `Reranker` — все за абстракциями |
| **Конфигурация через окружение** | `config.py` читает `.env`, создаёт компоненты через фабрики |
| **Асинхронность** | FastAPI + asyncio, фоновые задачи через `asyncio.create_task` |
| **Retry при сбоях** | `tenacity` в `vector_store/adapters.py` — до 3 попыток с backoff |
| **Наблюдаемость** | Prometheus Counter/Histogram + OpenTelemetry spans на каждом слое |
