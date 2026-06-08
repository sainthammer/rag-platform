"""RAG-пайплайн: поиск релевантных документов + генерация ответа.

Каждый вызов ``run()`` создаёт иерархию OTel-спанов::

    rag.query                        ← корневой спан (query, latency_ms)
    ├── rag.retrieve                 ← embed + vector_db.search (chunk_count, latency_ms)
    └── rag.generate                 ← llm.complete (model, latency_ms)

Prometheus-метрики обновляются после каждого успешного/неудачного вызова:
    rag_requests_total{status}
    rag_latency_seconds
    retrieval_chunk_count
    llm_requests_total{provider, model, status}
    llm_latency_seconds{provider, model}

Поток данных:
    query (str)
      │
      ├─▶ embed_fn(query)              → вектор запроса
      │
      ├─▶ vector_db.search(embedding)  → список документов (чанков)
      │
      ├─▶ budget.fit_chunks(documents) → усечённый набор, не превышающий
      │                                   токенный бюджет модели
      │
      ├─▶ template.format_system(ctx)  → системный промпт с контекстом
      │   template.format_user(query)  → вопрос (возможно, с инструкцией)
      │
      └─▶ llm.complete(messages)       → str | AsyncGenerator[str, None]
"""

import time
from typing import AsyncGenerator, Callable

from opentelemetry import trace

from vector_store.ports import VectorDB

from .llm_dataclasses import Message
from .ports import LLMProvider
from .prompt_templates import BASE, PromptTemplate
from .token_budget import TokenBudgetManager

_tracer = trace.get_tracer("rag-platform.pipeline")


class RAGPipeline:
    """Оркестратор полного RAG-цикла: retrieval → augmentation → generation.

    Класс не привязан к конкретному провайдеру или базе данных — принимает
    их через интерфейсы (``LLMProvider``, ``VectorDB``), что упрощает тесты
    и замену компонентов.

    Args:
        llm: Провайдер языковой модели (``OpenAIProvider``, ``AnthropicProvider``
            и т.д.).
        vector_db: Векторное хранилище, реализующее ``VectorDB.search``.
        embed_fn: Функция получения эмбеддинга для запроса пользователя.
            Принимает строку, возвращает вектор ``list[float]``. Намеренно
            sync-callable — embeddings-модуль ещё не реализован; при переходе
            к async-версии достаточно будет изменить только это место.
        template: Шаблон промпта (из ``prompt_templates``). По умолчанию BASE.
        n_results: Сколько ближайших чанков запрашивать из векторного хранилища.
            Реальный контекст может быть короче — ``TokenBudgetManager``
            отсеет лишние чанки по числу токенов.
        budget: Менеджер токенного бюджета. Если не задан — создаётся с
            параметрами по умолчанию (gpt-4o, 1000 зарезервированных токенов).
    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_db: VectorDB,
        embed_fn: Callable[[str], list[float]],
        template: PromptTemplate = BASE,
        n_results: int = 5,
        budget: TokenBudgetManager | None = None,
    ) -> None:
        self.llm = llm
        self.vector_db = vector_db
        self.embed_fn = embed_fn
        self.template = template
        self.n_results = n_results
        # Если бюджет не передан — инициализируем дефолтным, а не оставляем None,
        # чтобы в run() не было проверок на None.
        self.budget = budget or TokenBudgetManager()

    async def run(
        self,
        query: str,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Выполнить полный RAG-цикл для пользовательского запроса.

        Создаёт иерархию OTel-спанов:
            ``rag.query`` (корневой) → ``rag.retrieve`` → ``rag.generate``

        Атрибуты спанов:
            ``rag.query``:    query (первые 200 символов), latency_ms
            ``rag.retrieve``: query, chunk_count, n_results_requested, latency_ms
            ``rag.generate``: model, provider, latency_ms

        Обновляет Prometheus-метрики после завершения цикла.

        Args:
            query: Вопрос или инструкция пользователя на естественном языке.
            stream: Если ``True`` — вернуть генератор токенов для потоковой
                передачи ответа в UI. При стриминге ``rag.generate.latency_ms``
                фиксирует время до получения первого токена, а не до конца ответа.

        Returns:
            Полный ответ модели (``str``) при ``stream=False``,
            ``AsyncGenerator[str, None]`` при ``stream=True``.
        """
        from observability.metrics import (
            LLM_LATENCY_SECONDS,
            LLM_REQUESTS_TOTAL,
            RAG_LATENCY_SECONDS,
            RAG_REQUESTS_TOTAL,
            RETRIEVAL_CHUNK_COUNT,
        )

        provider = type(self.llm).__name__.replace("Provider", "").lower()
        model = getattr(self.llm, "model", "unknown")

        t_total = time.perf_counter()
        with _tracer.start_as_current_span("rag.query") as root_span:
            root_span.set_attribute("rag.query", query[:200])

            try:
                # 1–3. Retrieval: embed → search → token-budget filter
                t_retrieve = time.perf_counter()
                with _tracer.start_as_current_span("rag.retrieve") as span:
                    span.set_attribute("rag.query", query[:200])
                    span.set_attribute("rag.n_results_requested", self.n_results)

                    embedding = self.embed_fn(query)
                    results = self.vector_db.search(embedding, n_results=self.n_results)
                    chunks = self.budget.fit_chunks(results.documents)

                    span.set_attribute("rag.chunk_count", len(chunks))
                    span.set_attribute(
                        "rag.retrieve.latency_ms",
                        round((time.perf_counter() - t_retrieve) * 1000, 2),
                    )

                RETRIEVAL_CHUNK_COUNT.observe(len(chunks))

                # 4–5. Augmentation: собираем контекст и промпт
                context = "\n\n---\n\n".join(chunks)
                messages = [
                    Message(role="system", content=self.template.format_system(context)),
                    Message(role="user", content=self.template.format_user(query)),
                ]

                # 6. Generation: вызов LLM
                t_generate = time.perf_counter()
                with _tracer.start_as_current_span("rag.generate") as span:
                    span.set_attribute("rag.model", model)
                    span.set_attribute("rag.provider", provider)

                    try:
                        result = await self.llm.complete(messages, stream=stream)
                        generate_latency = time.perf_counter() - t_generate
                        span.set_attribute(
                            "rag.generate.latency_ms",
                            round(generate_latency * 1000, 2),
                        )
                        LLM_REQUESTS_TOTAL.labels(
                            provider=provider, model=model, status="success"
                        ).inc()
                        LLM_LATENCY_SECONDS.labels(
                            provider=provider, model=model
                        ).observe(generate_latency)

                    except Exception as exc:
                        LLM_REQUESTS_TOTAL.labels(
                            provider=provider, model=model, status="error"
                        ).inc()
                        span.record_exception(exc)
                        raise

                total_latency = time.perf_counter() - t_total
                root_span.set_attribute(
                    "rag.latency_ms", round(total_latency * 1000, 2)
                )
                RAG_REQUESTS_TOTAL.labels(status="success").inc()
                RAG_LATENCY_SECONDS.observe(total_latency)

                return result

            except Exception as exc:
                RAG_REQUESTS_TOTAL.labels(status="error").inc()
                root_span.record_exception(exc)
                raise

    async def run_detailed(self, query: str) -> tuple[str, list[str]]:
        """Выполнить RAG-цикл и вернуть ответ вместе с retrieved-чанками.

        Идентично ``run(stream=False)``, но дополнительно возвращает список
        чанков, которые были переданы в LLM как контекст. Используется
        eval_runner'ом: RAGAS требует ``retrieved_contexts`` для расчёта
        метрик ``ContextPrecision`` и ``ContextRecall``.

        Создаёт те же OTel-спаны и обновляет те же Prometheus-метрики,
        что и ``run()``.

        Args:
            query: Вопрос пользователя.

        Returns:
            Кортеж ``(answer: str, contexts: list[str])``.
        """
        from observability.metrics import (
            LLM_LATENCY_SECONDS,
            LLM_REQUESTS_TOTAL,
            RAG_LATENCY_SECONDS,
            RAG_REQUESTS_TOTAL,
            RETRIEVAL_CHUNK_COUNT,
        )

        provider = type(self.llm).__name__.replace("Provider", "").lower()
        model = getattr(self.llm, "model", "unknown")

        t_total = time.perf_counter()
        with _tracer.start_as_current_span("rag.query") as root_span:
            root_span.set_attribute("rag.query", query[:200])

            try:
                t_retrieve = time.perf_counter()
                with _tracer.start_as_current_span("rag.retrieve") as span:
                    span.set_attribute("rag.query", query[:200])
                    span.set_attribute("rag.n_results_requested", self.n_results)

                    embedding = self.embed_fn(query)
                    results = self.vector_db.search(embedding, n_results=self.n_results)
                    chunks = self.budget.fit_chunks(results.documents)

                    span.set_attribute("rag.chunk_count", len(chunks))
                    span.set_attribute(
                        "rag.retrieve.latency_ms",
                        round((time.perf_counter() - t_retrieve) * 1000, 2),
                    )

                RETRIEVAL_CHUNK_COUNT.observe(len(chunks))

                context = "\n\n---\n\n".join(chunks)
                messages = [
                    Message(role="system", content=self.template.format_system(context)),
                    Message(role="user", content=self.template.format_user(query)),
                ]

                t_generate = time.perf_counter()
                with _tracer.start_as_current_span("rag.generate") as span:
                    span.set_attribute("rag.model", model)
                    span.set_attribute("rag.provider", provider)

                    try:
                        result = await self.llm.complete(messages, stream=False)
                        generate_latency = time.perf_counter() - t_generate
                        span.set_attribute(
                            "rag.generate.latency_ms",
                            round(generate_latency * 1000, 2),
                        )
                        LLM_REQUESTS_TOTAL.labels(
                            provider=provider, model=model, status="success"
                        ).inc()
                        LLM_LATENCY_SECONDS.labels(
                            provider=provider, model=model
                        ).observe(generate_latency)
                    except Exception as exc:
                        LLM_REQUESTS_TOTAL.labels(
                            provider=provider, model=model, status="error"
                        ).inc()
                        span.record_exception(exc)
                        raise

                total_latency = time.perf_counter() - t_total
                root_span.set_attribute(
                    "rag.latency_ms", round(total_latency * 1000, 2)
                )
                RAG_REQUESTS_TOTAL.labels(status="success").inc()
                RAG_LATENCY_SECONDS.observe(total_latency)

                return str(result), chunks

            except Exception as exc:
                RAG_REQUESTS_TOTAL.labels(status="error").inc()
                root_span.record_exception(exc)
                raise
