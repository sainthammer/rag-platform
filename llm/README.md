# Модуль `llm/` — провайдеры и RAG-пайплайн

Модуль отвечает за взаимодействие с языковыми моделями и оркестрацию полного RAG-цикла:
поиск релевантных документов → формирование контекста → генерация ответа.

---

## Структура файлов

```
llm/
├── llm_dataclasses.py   # Тип данных Message
├── ports.py             # Абстрактный интерфейс LLMProvider
├── adapters.py          # Реализации: OpenAI, Anthropic, Ollama
├── prompt_templates.py  # Шаблоны системных промптов
├── token_budget.py      # Управление токенным бюджетом
├── pipeline.py          # Главный оркестратор RAGPipeline
└── example.py           # Запускаемый smoke-тест
```

---

## Точка входа — `config.py`

При старте приложения модуль `config.py` создаёт объект `settings`, который читает
переменные окружения из файла `.env`:

```
LLM_PROVIDER=ollama   →   settings.llm_provider = "ollama"
LLM_MODEL=llama3.2    →   settings.llm_model    = "llama3.2"
```

Когда нужен провайдер — вызывается фабрика `build_llm_provider()`:

```python
from config import build_llm_provider

llm = build_llm_provider()
# читает settings.llm_provider → создаёт OllamaProvider(model="llama3.2")
```

Доступные значения `LLM_PROVIDER`: `openai`, `anthropic`, `ollama`.

---

## Слой абстракции — `ports.py` и `llm_dataclasses.py`

### `llm_dataclasses.py` — единый тип сообщения

Все провайдеры общаются через один тип данных:

```python
Message(role="system",    content="Ты полезный ассистент...")
Message(role="user",      content="Что такое RAG?")
Message(role="assistant", content="RAG — это...")  # для multi-turn диалога
```

### `ports.py` — контракт провайдера

Определяет **единственный метод**, который обязан реализовать любой провайдер:

```python
class LLMProvider(ABC):
    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]: ...
```

- `stream=False` → дождаться полного ответа, вернуть `str`
- `stream=True`  → вернуть `AsyncGenerator`, токены приходят по мере генерации

Весь остальной код (`RAGPipeline`, тесты) работает только через этот интерфейс —
конкретный провайдер можно поменять, не трогая бизнес-логику.

---

## Реализации провайдеров — `adapters.py`

### `OpenAIProvider`

```python
provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-...")
```

Использует `AsyncOpenAI` из официального SDK. Если `api_key=None` — SDK читает
переменную окружения `OPENAI_API_KEY` самостоятельно.
Параметр `base_url` позволяет подключиться к прокси или Azure OpenAI.

### `AnthropicProvider`

```python
provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="sk-ant-...")
```

Протокол Anthropic отличается от OpenAI: системный промпт передаётся **отдельным полем**
`system=`, а не первым элементом списка сообщений. Метод `_split_messages()` выполняет
это разделение автоматически перед каждым запросом.

Параметр `max_tokens` обязателен для Anthropic API (в отличие от OpenAI).

### `OllamaProvider`

```python
provider = OllamaProvider(model="llama3.2", base_url="http://localhost:11434/v1")
```

Ollama поднимает OpenAI-совместимый REST API, поэтому используется тот же `AsyncOpenAI`-клиент
с переопределённым `base_url`. Новых зависимостей не требует.
`api_key="ollama"` — произвольная строка, SDK требует непустое значение, Ollama его игнорирует.

> Для стриминга все провайдеры используют `create(stream=True)` и итерируют
> `chunk.choices[0].delta.content` — это работает с любым OpenAI-совместимым endpoint'ом.

---

## Шаблоны промптов — `prompt_templates.py`

Шаблон — неизменяемый объект с системным промптом и дыркой `{context}` под реальные данные:

```python
BASE.format_system("Векторные БД хранят эмбеддинги...")
# → "Ты полезный ассистент...\n\nКонтекст:\nВекторные БД хранят эмбеддинги..."

STRICT.format_user("Что такое RAG?")
# → "Отвечай строго по контексту. Не придумывай.\n\nЧто такое RAG?"
```

### Готовые шаблоны

| Константа      | Когда использовать                                    |
|----------------|-------------------------------------------------------|
| `BASE`         | Обычные вопросы, модель может дополнять из памяти     |
| `STRICT`       | Нельзя выходить за пределы контекста, ноль галлюцинаций |
| `CITATION`     | Нужны ссылки на источники вида [1], [2]               |
| `MULTILINGUAL` | Пользователь пишет на разных языках                   |

Получить шаблон по имени (удобно при конфигурировании через env):

```python
from llm.prompt_templates import get_template

template = get_template("strict")  # → STRICT
```

---

## Токенный бюджет — `token_budget.py`

У каждой модели есть лимит токенов на один запрос (контекстное окно). Если найденных
чанков слишком много — они не влезут. `TokenBudgetManager` решает эту проблему:

```
Окно модели (например, 128 000 токенов)
  ├── reserved_tokens (1 000) — место под промпт, вопрос и ответ
  └── budget (127 000)        — место под контекст из чанков
```

```python
budget = TokenBudgetManager(model="gpt-4o", reserved_tokens=1000)

budget.fit_chunks(["чанк1 (500 токенов)", "чанк2 (300)", "чанк3 (60 000)"])
# → ["чанк1", "чанк2"]   ← чанк3 не влезает, отброшен
```

Чанки обрабатываются в исходном порядке (по убыванию релевантности), поэтому
самые важные всегда попадают в контекст первыми.

---

## Главный оркестратор — `pipeline.py`

`RAGPipeline` соединяет все компоненты и выполняет полный RAG-цикл за 6 шагов:

```
query = "Что такое RAG?"
         │
         ▼  1. Векторизуем запрос
embed_fn(query) → [0.9, 0.1, 0.0, ...]

         │
         ▼  2. Ищем похожие документы
vector_db.search(embedding, n_results=5)
→ SearchResult(documents=["RAG это...", "Векторные БД...", ...])

         │
         ▼  3. Отбрасываем чанки, которые не влезают в бюджет
budget.fit_chunks(documents)
→ ["RAG это...", "Векторные БД..."]

         │
         ▼  4. Собираем текст контекста
context = "RAG это...\n\n---\n\nВекторные БД..."

         │
         ▼  5. Формируем сообщения
messages = [
  Message("system", template.format_system(context)),
  Message("user",   template.format_user(query)),
]

         │
         ▼  6. Отправляем запрос модели
llm.complete(messages, stream=False)
→ "RAG (Retrieval-Augmented Generation) — это подход, при котором..."
```

### Создание пайплайна

```python
from llm.pipeline import RAGPipeline
from llm.prompt_templates import STRICT
from llm.token_budget import TokenBudgetManager
from config import build_llm_provider
from vector_store.adapters import QdrantDB

pipeline = RAGPipeline(
    llm=build_llm_provider(),          # провайдер из .env
    vector_db=QdrantDB("docs", ...),   # векторное хранилище
    embed_fn=my_embed_function,        # функция: str → list[float]
    template=STRICT,                   # шаблон промпта
    n_results=5,                       # сколько чанков запрашивать
    budget=TokenBudgetManager(),       # токенный бюджет
)

# Обычный запрос
answer = await pipeline.run("Как работает чанкинг?")

# Стриминг
async for chunk in await pipeline.run("Как работает чанкинг?", stream=True):
    print(chunk, end="", flush=True)
```

---

## Схема зависимостей

```
config.py
  └── build_llm_provider()
        └── adapters.py  (OpenAIProvider / AnthropicProvider / OllamaProvider)
              └── ports.py (LLMProvider — интерфейс)
                    └── llm_dataclasses.py (Message)

pipeline.py (RAGPipeline)
  ├── ports.py               ← любой LLMProvider
  ├── vector_store/ports.py  ← любая VectorDB (ChromaDB / QdrantDB)
  ├── prompt_templates.py    ← шаблон промпта
  └── token_budget.py        ← менеджер бюджета
```

`RAGPipeline` зависит только от **интерфейсов**, а не от конкретных классов.
Замена `OllamaProvider` на `OpenAIProvider` или `QdrantDB` на `ChromaDB`
не требует изменений в `pipeline.py`.
