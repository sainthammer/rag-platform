"""EmbeddingService — получение текстовых эмбеддингов с OTel-трейсингом и Prometheus-метриками.

Каждый вызов ``embed`` / ``embed_batch`` создаёт OTel-span с атрибутами
``embedding.model``, ``embedding.input_length``, ``embedding.latency_ms``
и обновляет Prometheus-счётчики и гистограммы из ``observability/metrics.py``.

Интеграция с RAGPipeline:
    EmbeddingService возвращает async-методы. Для передачи в RAGPipeline
    (который принимает sync ``embed_fn``) используйте asyncio.run() или
    оборачивайте через run_sync-адаптер в точке сборки приложения.
"""

import time
from typing import TYPE_CHECKING

from opentelemetry import trace

if TYPE_CHECKING:
    pass

_tracer = trace.get_tracer("rag-platform.embeddings")


class EmbeddingService:
    """Сервис векторизации текстов через OpenAI Embeddings API.

    Автоматически записывает OTel-spans и обновляет Prometheus-метрики
    для каждого запроса к API.

    Args:
        model: Идентификатор модели OpenAI Embeddings
            (например, ``text-embedding-3-small``).
        api_key: API-ключ OpenAI. Если ``None``, SDK читает
            переменную окружения ``OPENAI_API_KEY``.
        base_url: Базовый URL API. Полезно для OpenAI-совместимых
            провайдеров (Azure, локальный прокси и т.д.).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, text: str) -> list[float]:
        """Получить вектор эмбеддинга для одного текста.

        Создаёт OTel-span ``embedding.embed`` с атрибутами:
            - ``embedding.model`` — имя модели
            - ``embedding.input_length`` — длина входной строки в символах
            - ``embedding.latency_ms`` — задержка запроса в миллисекундах
            - ``embedding.dimension`` — размерность полученного вектора

        Args:
            text: Строка для векторизации.

        Returns:
            Вектор эмбеддинга ``list[float]`` размерности модели.

        Raises:
            openai.OpenAIError: При ошибке API (неверный ключ, rate limit и т.д.).
        """
        from observability.metrics import EMBEDDING_LATENCY_SECONDS, EMBEDDING_REQUESTS_TOTAL

        t0 = time.perf_counter()
        with _tracer.start_as_current_span("embedding.embed") as span:
            span.set_attribute("embedding.model", self.model)
            span.set_attribute("embedding.input_length", len(text))

            try:
                response = await self.client.embeddings.create(
                    input=text, model=self.model
                )
                embedding = response.data[0].embedding

                latency = time.perf_counter() - t0
                span.set_attribute("embedding.latency_ms", round(latency * 1000, 2))
                span.set_attribute("embedding.dimension", len(embedding))

                EMBEDDING_REQUESTS_TOTAL.labels(model=self.model, status="success").inc()
                EMBEDDING_LATENCY_SECONDS.labels(model=self.model).observe(latency)

                return embedding

            except Exception as exc:
                EMBEDDING_REQUESTS_TOTAL.labels(model=self.model, status="error").inc()
                span.record_exception(exc)
                raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для нескольких текстов одним API-запросом.

        Эффективнее последовательных вызовов ``embed`` при большом числе текстов,
        так как все тексты отправляются в одном HTTP-запросе.

        Создаёт OTel-span ``embedding.embed_batch`` с атрибутами:
            - ``embedding.model`` — имя модели
            - ``embedding.batch_size`` — число входных строк
            - ``embedding.latency_ms`` — задержка запроса в миллисекундах

        Args:
            texts: Список строк для векторизации.

        Returns:
            Список векторов ``list[list[float]]`` в том же порядке, что входные тексты.

        Raises:
            openai.OpenAIError: При ошибке API.
        """
        from observability.metrics import EMBEDDING_LATENCY_SECONDS, EMBEDDING_REQUESTS_TOTAL

        t0 = time.perf_counter()
        with _tracer.start_as_current_span("embedding.embed_batch") as span:
            span.set_attribute("embedding.model", self.model)
            span.set_attribute("embedding.batch_size", len(texts))

            try:
                response = await self.client.embeddings.create(
                    input=texts, model=self.model
                )
                embeddings = [item.embedding for item in response.data]

                latency = time.perf_counter() - t0
                span.set_attribute("embedding.latency_ms", round(latency * 1000, 2))

                EMBEDDING_REQUESTS_TOTAL.labels(model=self.model, status="success").inc()
                EMBEDDING_LATENCY_SECONDS.labels(model=self.model).observe(latency)

                return embeddings

            except Exception as exc:
                EMBEDDING_REQUESTS_TOTAL.labels(model=self.model, status="error").inc()
                span.record_exception(exc)
                raise
