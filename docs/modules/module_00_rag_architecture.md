# Модуль 0 — Что такое RAG и архитектура проекта

## Зачем нужен RAG

**RAG (Retrieval-Augmented Generation)** решает главную проблему языковых моделей: они знают только то, на чём обучались. Если спросить GPT-4 о внутренних документах компании или свежих данных — модель либо ответит неправильно, либо скажет «не знаю».

RAG позволяет «подкармливать» модель актуальными данными из твоей базы знаний прямо во время ответа.

## Три шага RAG

```
1. Retrieve  — найти в базе документы, похожие на вопрос
2. Augment   — вставить эти документы в промпт как контекст
3. Generate  — LLM отвечает, опираясь на контекст (а не на обучение)
```

## Как модули связаны

### Путь вопроса

```
POST /v1/ask
     │
     ▼
[api] require_auth + RequestIDMiddleware + rate limiter
     │
     ▼
[retrieval/pipeline.py] RAGPipeline.ask()
     │
     ├──▶ [embeddings] embed_fn(question)   → вектор запроса
     │
     ├──▶ [vector_store] db.search()        → топ-N чанков
     │         (dense или hybrid dense+BM25)
     │
     ├──▶ [retrieval/rerankers] reranker    → пересортировка
     │         (CrossEncoder или MMR)
     │
     ├──▶ [llm/token_budget] fit_chunks()   → обрезка по токенам
     │
     ├──▶ [llm/prompt_templates] format_*() → сборка промпта
     │
     └──▶ [llm/adapters] llm.complete()    → ответ LLM
```

### Путь документа (ingest)

```
POST /v1/ingest
     │
     ▼
[chunking/ingest.py] ingest()
     │
     ├── определяем лоадер по расширению (.pdf → PDFLoader, .md → MarkdownLoader...)
     ├── loader.load(path) → текст
     └── chunker.split(text) → список Chunk
           (FixedSizeChunker / ByHeaderChunker / SemanticChunker)
     │
     ▼
[embeddings] embed_service.embed_batch(chunks) → векторы
     │
     ▼
[vector_store] db.add(ids, embeddings, documents, metadatas)
```

### Поддерживающая инфраструктура

```
[config.py]         единая конфигурация + фабрики компонентов
[observability/]    Prometheus метрики + OpenTelemetry трейсинг → Jaeger/Grafana
[evaluation/]       RAGAS оценка качества: Faithfulness, ContextRecall...
[mcp/]              MCP-сервер: 4 инструмента для Claude Desktop / AI-агентов
[tests/]            unit + integration + e2e тесты
```

## Архитектурный паттерн: Ports & Adapters

Весь проект построен на одном паттерне. Бизнес-логика зависит только от абстракций (Port), конкретные реализации (Adapters) подменяются без изменения кода.

```
Port (абстракция)       Adapters (реализации)
────────────────────    ──────────────────────────────────────────
DocumentLoader      →   TextLoader, MarkdownLoader, HTMLLoader, PDFLoader
Chunker             →   FixedSizeChunker, ByHeaderChunker, SemanticChunker
EmbeddingService    →   OpenAIEmbeddingService, SentenceTransformersService,
                        OllamaEmbeddingService, FakeEmbeddingService
VectorDB            →   ChromaDB, QdrantVectorStore, HybridVectorStore
LLMProvider         →   OpenAIProvider, AnthropicProvider, OllamaProvider
Reranker            →   CrossEncoderReranker, MMRReranker
```

Чтобы поменять провайдер — достаточно изменить одну строку в `.env`. Код не меняется.

## Структура файлов проекта

```
rag-platform/
  config.py               конфигурация + фабрики
  chunking/
    ports.py              Chunk, Chunker, DocumentLoader
    loaders.py            TextLoader, MarkdownLoader, HTMLLoader, PDFLoader
    adapters.py           FixedSizeChunker, ByHeaderChunker, SemanticChunker
    ingest.py             фасад: ingest(path, strategy, ...)
  embeddings/
    ports.py              EmbeddingService
    adapters.py           OpenAI, SentenceTransformers, Ollama, Fake
    cache.py              SQLite-кэш векторов
    service.py            фабрика с кэшем
  vector_store/
    ports.py              VectorDB
    adapters.py           ChromaDB, QdrantVectorStore, HybridVectorStore
    bm25.py               BM25SparseVectorizer
    store_dataclasses.py  SearchResult, CollectionStats
  retrieval/
    pipeline.py           RAGPipeline (полный RAG-цикл)
    retriever.py          Retriever (только поиск)
    ports.py              Reranker
    rerankers.py          CrossEncoderReranker, MMRReranker
  llm/
    ports.py              LLMProvider
    adapters.py           OpenAI, Anthropic, Ollama
    prompt_templates.py   BASE, STRICT, CITATION, MULTILINGUAL
    token_budget.py       TokenBudgetManager
  api/
    app.py                FastAPI + lifespan + роутеры
    limiter.py            rate limiting (slowapi)
    middleware/
      auth.py             X-API-Key + JWT Bearer
      request_id.py       X-Request-ID middleware
    routers/              ask, ingest, search, eval, collections, health, metrics
    schemas.py            Pydantic-модели запросов/ответов
  evaluation/
    ragas_eval.py         RAGAS метрики
    eval_runner.py        запуск оценки + HTML-отчёт
    testcase.py           TestCase датакласс
    testcases_dataset.py  набор тест-кейсов
  observability/
    metrics.py            Prometheus Counter/Histogram
    tracing.py            OpenTelemetry + FastAPI инструментация
  mcp/
    rag_server.py         MCP-сервер: list_collections, search, ingest, ask
  tests/
    unit/                 изолированные тесты отдельных модулей
    integration/          тесты нескольких модулей вместе
    e2e/                  HTTP API как чёрный ящик
  Dockerfile
  docker-compose.yml      api, chroma, qdrant, ollama, jaeger, prometheus, grafana
```
