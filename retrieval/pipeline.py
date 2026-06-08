"""RAG-пайплайн с enriched-ответом, streaming и параллельными коллекциями.

Поток данных для ask():
    question (str)
      │
      ├─▶ embed_fn(question)              → вектор запроса
      ├─▶ vector_db_factory(collection)   → VectorDB для нужной коллекции
      ├─▶ db.search(embedding)            → SearchResult (chunks + distances)
      │
      ├─▶ fallback, если нет чанков или confidence < threshold
      │
      ├─▶ budget.fit_chunks(docs)         → усечённый контекст
      ├─▶ template.format_*()             → промпт
      └─▶ llm.complete(messages)          → answer → RAGResponse
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from llm.llm_dataclasses import Message
from llm.ports import LLMProvider
from llm.prompt_templates import BASE, PromptTemplate
from llm.token_budget import TokenBudgetManager
from vector_store.ports import VectorDB

FALLBACK_ANSWER = "К сожалению, я не нашёл информации по вашему вопросу."


@dataclass
class SourceChunk:
    """Один извлечённый фрагмент документа с оценкой релевантности."""

    text: str
    score: float  # [0, 1]: 1 — идеальное совпадение, 0 — максимальная удалённость
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Структурированный ответ RAG-пайплайна."""

    answer: str
    sources: list[SourceChunk]
    confidence: float  # max score среди найденных чанков, [0, 1]
    latency_ms: float


class RAGPipeline:
    """RAG-пайплайн: retrieval → augmentation → generation с поддержкой:
    - структурированных ответов (RAGResponse с источниками и confidence);
    - streaming через ask_stream();
    - fallback при отсутствии релевантного контекста;
    - параллельного поиска по нескольким коллекциям через ask_multi().

    Args:
        llm: Провайдер языковой модели.
        vector_db_factory: Фабрика ``collection_name → VectorDB``. Вызывается
            на каждый ask(), что позволяет маршрутизировать по разным
            коллекциям без предварительного создания всех экземпляров.
        embed_fn: Синхронная функция векторизации запроса.
        template: Шаблон промпта. По умолчанию BASE.
        n_results: Сколько ближайших чанков запрашивать у VectorDB.
        budget: Менеджер токенного бюджета. Если не задан — создаётся дефолтный.
        score_threshold: Минимальный score лучшего чанка. Если confidence
            ниже порога или нет ни одного чанка — возвращается fallback_answer
            без вызова LLM.
        fallback_answer: Текст ответа при отсутствии релевантного контекста.
    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_db_factory: Callable[[str], VectorDB],
        embed_fn: Callable[[str], list[float]],
        template: PromptTemplate = BASE,
        n_results: int = 5,
        budget: TokenBudgetManager | None = None,
        score_threshold: float = 0.0,
        fallback_answer: str = FALLBACK_ANSWER,
    ) -> None:
        self.llm = llm
        self.vector_db_factory = vector_db_factory
        self.embed_fn = embed_fn
        self.template = template
        self.n_results = n_results
        self.budget = budget or TokenBudgetManager()
        self.score_threshold = score_threshold
        self.fallback_answer = fallback_answer

    async def ask(
        self,
        question: str,
        collection: str,
        stream: bool = False,
    ) -> RAGResponse:
        """Выполнить полный RAG-цикл и вернуть структурированный ответ.

        При ``stream=True`` LLM-вызов выполняется в потоковом режиме,
        но ответ накапливается в строку — ``RAGResponse.answer`` всегда полный.
        Для побайтовой выдачи токенов используйте ask_stream().

        Args:
            question: Вопрос пользователя.
            collection: Имя коллекции в векторном хранилище.
            stream: Использовать ли streaming в LLM-вызове.

        Returns:
            RAGResponse с ответом, источниками, уровнем уверенности и задержкой.
        """
        start = time.monotonic()

        sources, confidence = self._retrieve(question, collection)

        if not sources or confidence < self.score_threshold:
            return RAGResponse(
                answer=self.fallback_answer,
                sources=[],
                confidence=confidence,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        messages = self._build_messages(question, sources)
        llm_result = await self.llm.complete(messages, stream=stream)

        if isinstance(llm_result, str):
            answer = llm_result
        else:
            parts: list[str] = []
            async for token in llm_result:
                parts.append(token)
            answer = "".join(parts)

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def ask_stream(
        self,
        question: str,
        collection: str,
    ) -> AsyncGenerator[str, None]:
        """Стриминг ответа токен за токеном.

        При отсутствии релевантных чанков выдаёт fallback одним блоком.
        Вызывается без await — итерируется через ``async for``:

            async for token in pipeline.ask_stream("вопрос", "коллекция"):
                print(token, end="", flush=True)

        Args:
            question: Вопрос пользователя.
            collection: Имя коллекции в векторном хранилище.

        Yields:
            Текстовые фрагменты ответа по мере генерации.
        """
        sources, confidence = self._retrieve(question, collection)

        if not sources or confidence < self.score_threshold:
            yield self.fallback_answer
            return

        messages = self._build_messages(question, sources)
        llm_result = await self.llm.complete(messages, stream=True)

        if isinstance(llm_result, str):
            yield llm_result
        else:
            async for token in llm_result:
                yield token

    async def ask_multi(
        self,
        question: str,
        collections: list[str],
    ) -> list[RAGResponse]:
        """Запросить несколько коллекций параллельно через asyncio.gather.

        Args:
            question: Вопрос пользователя.
            collections: Список имён коллекций.

        Returns:
            Список RAGResponse в порядке, соответствующем входному списку collections.
        """
        tasks = [self.ask(question, collection) for collection in collections]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Приватные вспомогательные методы
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, collection: str) -> tuple[list[SourceChunk], float]:
        """Выполнить embedding + vector search и вернуть источники с confidence.

        Returns:
            Кортеж (список SourceChunk, max score среди них).
        """
        embedding = self.embed_fn(question)
        db = self.vector_db_factory(collection)
        result = db.search(embedding, n_results=self.n_results)

        sources = [
            SourceChunk(
                text=doc,
                score=_distance_to_score(dist),
                doc_id=doc_id,
                metadata=meta,
            )
            for doc_id, doc, dist, meta in zip(
                result.ids, result.documents, result.distances, result.metadatas
            )
        ]
        confidence = max((s.score for s in sources), default=0.0)
        return sources, confidence

    def _build_messages(self, question: str, sources: list[SourceChunk]) -> list[Message]:
        """Собрать список сообщений для LLM из источников и вопроса."""
        chunks = self.budget.fit_chunks([s.text for s in sources])
        context = "\n\n---\n\n".join(chunks)
        return [
            Message(role="system", content=self.template.format_system(context)),
            Message(role="user", content=self.template.format_user(question)),
        ]


def _distance_to_score(distance: float) -> float:
    """Преобразовать дистанцию (меньше = лучше) в score (больше = лучше).

    Формула ``1 / (1 + distance)`` отображает [0, ∞) в (0, 1]:
      distance = 0   → score = 1.0  (идентичные векторы)
      distance = 1   → score = 0.5
      distance = 9   → score = 0.1
    """
    return 1.0 / (1.0 + distance)
