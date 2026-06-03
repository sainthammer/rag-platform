"""RAG-пайплайн: поиск релевантных документов + генерация ответа.

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

from typing import AsyncGenerator, Callable

from vector_store.ports import VectorDB

from .llm_dataclasses import Message
from .ports import LLMProvider
from .prompt_templates import BASE, PromptTemplate
from .token_budget import TokenBudgetManager


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

        Args:
            query: Вопрос или инструкция пользователя на естественном языке.
            stream: Если ``True`` — вернуть генератор токенов для потоковой
                передачи ответа в UI.

        Returns:
            Полный ответ модели (``str``) при ``stream=False``,
            ``AsyncGenerator[str, None]`` при ``stream=True``.
        """
        # 1. Получаем вектор запроса для семантического поиска.
        embedding = self.embed_fn(query)

        # 2. Ищем n_results ближайших чанков в векторном хранилище.
        results = self.vector_db.search(embedding, n_results=self.n_results)

        # 3. Отбрасываем чанки, которые не вмещаются в токенный бюджет.
        #    Чанки упорядочены по релевантности — самые важные идут первыми.
        chunks = self.budget.fit_chunks(results.documents)

        # 4. Соединяем чанки разделителем, который модель видит как границу
        #    между разными фрагментами источника.
        context = "\n\n---\n\n".join(chunks)

        # 5. Строим диалог: системный промпт с контекстом + вопрос пользователя.
        messages = [
            Message(role="system", content=self.template.format_system(context)),
            Message(role="user", content=self.template.format_user(query)),
        ]

        # 6. Отправляем запрос модели; результат — str или AsyncGenerator.
        return await self.llm.complete(messages, stream=stream)
