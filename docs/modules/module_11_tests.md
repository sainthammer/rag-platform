# Модуль 11 — tests/

**Файлы:** `tests/unit/`, `tests/integration/`, `tests/e2e/`

Три уровня тестирования: unit, integration, e2e.

## Структура

```
tests/
  unit/
    test_embeddings.py         тесты провайдеров эмбеддингов
    test_embedding_cache.py    тесты SQLite-кэша
    test_pipeline.py           тесты RAGPipeline
    test_rag_cache_and_logging.py  тесты кэша + логирования
    test_chunking.py           тесты лоадеров и стратегий чанкинга
    test_vector_store.py       тесты ChromaDB/Qdrant (mock/in-memory)
    test_llm_adapters.py       тесты LLM-провайдеров
    test_evaluation.py         тесты RAGAS eval
    test_observability.py      тесты метрик и трейсинга
  integration/
    test_rag_pipeline.py       полный пайплайн с реальной Qdrant in-memory
    test_ingest_retrieve.py    ingest → search round-trip
    test_chroma_qdrant.py      сравнение бэкендов
  e2e/
    test_api_flow.py           HTTP API как чёрный ящик
```

## Три уровня тестов

### Unit-тесты — изоляция

Тестируют один модуль, все зависимости заменены заглушками. Работают без Docker, без сети, без API-ключей.

**Ключевой инструмент — `FakeEmbeddingService`:**
```python
def test_embed_batch():
    svc = FakeEmbeddingService(size=8)
    vecs = svc.embed_batch(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 8
    assert vecs[0] == vecs[0]   # детерминированный
```

**Тест кэша эмбеддингов:**
```python
def test_cache_hit():
    fake = FakeEmbeddingService(size=8)
    cache = EmbeddingCache(path=":memory:")
    svc = CachedEmbeddingService(fake, cache)

    svc.embed("hello")   # промах — вычисляем
    svc.embed("hello")   # хит — берём из кэша
    assert len(fake.calls) == 1   # embed_fn вызван только один раз
```

**Тест RAGPipeline с заглушками:**
```python
async def test_ask_returns_fallback():
    fake_db = FakeVectorDB(results=[])   # пустая БД
    pipeline = RAGPipeline(
        llm=FakeLLM(),
        vector_db_factory=lambda col: fake_db,
        embed_fn=FakeEmbeddingService().embed,
        score_threshold=0.5,
    )
    response = await pipeline.ask("вопрос", "docs")
    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []
```

### Integration-тесты — реальные компоненты

Реальная Qdrant/Chroma in-memory + реальный `FakeEmbeddingService`. Нет сети, нет API-ключей, но логика работает как в production.

```python
@pytest.fixture
def vector_store():
    return QdrantVectorStore(
        collection="test",
        vector_size=8,
        in_memory=True,   # Qdrant в памяти, без Docker
    )

def test_ingest_and_retrieve(vector_store):
    fake_emb = FakeEmbeddingService(size=8)
    chunks = ["Python — язык программирования", "FastAPI — веб-фреймворк"]
    embeddings = fake_emb.embed_batch(chunks)
    vector_store.add(ids=["c0", "c1"], embeddings=embeddings, documents=chunks)

    query_vec = fake_emb.embed("Python")
    result = vector_store.search(query_vec, n_results=1)
    assert "Python" in result.documents[0]
```

### E2E-тесты — HTTP API

Запускают FastAPI через `httpx.AsyncClient` и проверяют HTTP-ответы. Полный стек без сети.

```python
@pytest.fixture
async def client(app):
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

async def test_ingest_and_ask(client):
    # Загружаем документ
    r = await client.post("/v1/ingest", data={"text": "Python — язык программирования"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # Ждём завершения
    await asyncio.sleep(0.1)
    r = await client.get(f"/v1/ingest/{job_id}")
    assert r.json()["status"] == "done"

    # Задаём вопрос
    r = await client.post("/v1/ask", json={"question": "что такое Python?"})
    assert r.status_code == 200
    assert "язык" in r.json()["answer"].lower()
```

## Паттерны тестирования

### Fixture-цепочки

Pytest fixtures создают слоёный контекст:
```python
@pytest.fixture
def settings():
    return Settings(vector_store_backend="qdrant", ...)

@pytest.fixture
def vector_db(settings):
    return build_vector_db(settings)   # зависит от settings

@pytest.fixture
def pipeline(vector_db, settings):
    return build_rag_pipeline(settings)   # зависит от vector_db
```

### Параметризация

Один тест запускается с несколькими наборами данных:
```python
@pytest.mark.parametrize("chunk_size,expected_count", [
    (100, 10),
    (500, 3),
    (1000, 1),
])
def test_chunking_count(chunk_size, expected_count, long_text):
    chunks = chunk_text(long_text, chunk_size=chunk_size)
    assert len(chunks) == expected_count
```

### Mocking

Внешние API мокаются через `unittest.mock` или `pytest-mock`:
```python
def test_openai_provider(mocker):
    mock_client = mocker.patch("openai.AsyncOpenAI")
    mock_client.return_value.chat.completions.create.return_value = fake_response
    provider = OpenAIProvider(api_key="test")
    result = await provider.complete([Message(role="user", content="hello")])
    assert result == "fake answer"
```

## Запуск тестов

```bash
# Все тесты
pytest

# Только unit
pytest tests/unit/

# С покрытием
pytest --cov=. --cov-report=html

# По маркеру
pytest -m "not slow"

# Конкретный файл
pytest tests/unit/test_pipeline.py -v
```

## Coverage целевые показатели

- Unit-тесты: ≥ 90% покрытие изолированной логики
- Integration-тесты: ≥ 75% покрытие cross-module взаимодействий
- E2E: smoke-тесты основных user flow
