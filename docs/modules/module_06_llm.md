# Модуль 6 — llm/

**Файлы:** `llm/ports.py`, `llm/adapters.py`, `llm/prompt_templates.py`, `llm/token_budget.py`, `llm/llm_dataclasses.py`

Модуль отвечает за работу с языковыми моделями: промпты, бюджет токенов, адаптеры под разные LLM.

## Структура

```
llm/
  ports.py            абстракция LLMProvider
  adapters.py         OpenAI, Anthropic, Ollama
  prompt_templates.py BASE, STRICT, CITATION, MULTILINGUAL
  token_budget.py     TokenBudgetManager
  llm_dataclasses.py  Message датакласс
```

## llm_dataclasses.py — Message

```python
@dataclass
class Message:
    role: str      # "system", "user", "assistant"
    content: str   # текст сообщения
```

Внутреннее представление сообщений. Каждый адаптер конвертирует в формат своего SDK.

## ports.py — один метод

```python
class LLMProvider(ABC):
    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]: ...
```

- `stream=False` → дождаться полного ответа и вернуть `str`
- `stream=True` → вернуть `AsyncGenerator[str, None]`, каждый элемент — очередной токен

## adapters.py — три провайдера

### `OpenAIProvider`

```python
provider = OpenAIProvider(
    model="gpt-4o-mini",
    api_key="sk-...",
)
```

Использует `AsyncOpenAI` — асинхронный клиент. Оба режима (stream/non-stream) реализованы через `chat.completions.create`.

При `stream=True` возвращает внутренний async-генератор `_stream()`, который читает SSE-чанки от API:
```python
async def _stream():
    response = await client.chat.completions.create(model=..., messages=..., stream=True)
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

### `AnthropicProvider`

```python
provider = AnthropicProvider(
    model="claude-haiku-4-5-20251001",
    api_key="sk-ant-...",
    max_tokens=1024,    # Anthropic требует явного указания
)
```

**Ключевое отличие от OpenAI:** Anthropic API принимает системный промпт **отдельным параметром**, а не первым элементом `messages`. Метод `_split_messages()` выполняет разделение:

```python
# OpenAI:    messages=[{"role": "system", ...}, {"role": "user", ...}]
# Anthropic: system="...", messages=[{"role": "user", ...}]
```

При стриминге использует `client.messages.stream()` — контекстный менеджер с `text_stream`.

### `OllamaProvider`

```python
provider = OllamaProvider(
    model="llama3.2",
    base_url="http://localhost:11434/v1",
)
```

Ollama предоставляет **OpenAI-совместимый** REST API, поэтому используется тот же `AsyncOpenAI` клиент с другим `base_url`. Никаких дополнительных зависимостей.

Quirk: OpenAI SDK требует непустой `api_key`, поэтому передаётся заглушка `"ollama"`.

## prompt_templates.py — шаблоны промптов

### `PromptTemplate` — структура

```python
@dataclass(frozen=True)   # неизменяемый — используется как константа
class PromptTemplate:
    system: str       # системный промпт с плейсхолдером {context}
    user_prefix: str  # доп. инструкция перед вопросом (опционально)

    def format_system(self, context: str) -> str: ...   # подставляет {context}
    def format_user(self, query: str) -> str: ...       # добавляет user_prefix
```

### Четыре готовых шаблона

**`BASE`** — универсальный:
```
Ты полезный ассистент. Отвечай на вопрос, используя контекст.
Если контекста недостаточно — честно скажи об этом.
```

**`STRICT`** — только по контексту, без домысливания:
```
Ты точный ассистент. Отвечай ТОЛЬКО на основе контекста.
НЕ используй внешние знания. Если ответа нет — скажи об этом.
```
+ `user_prefix="Отвечай строго по контексту. Не придумывай."` — инструкция дублируется в user-туре. Повторение в двух ролях снижает вероятность галлюцинаций.

**`CITATION`** — каждое утверждение со ссылкой `[1]`, `[2]`:
```
Каждое утверждение сопровождай ссылкой [1], [2].
В конце — список «Источники:».
```

**`MULTILINGUAL`** — отвечает на языке вопроса:
```
Определи язык вопроса и отвечай на том же языке.
```

### Получение шаблона по имени

```python
template = get_template("strict")   # из строки, например из env-переменной
```

Используется в `RAGPipeline` при конфигурировании через API.

## token_budget.py — TokenBudgetManager

Каждая LLM имеет ограниченное контекстное окно. Если вставить слишком много чанков — запрос упадёт с ошибкой или лишний контекст будет срезан.

### Устройство бюджета

```
total context window (например 128 000 токенов для gpt-4o)
├── reserved_tokens (1 000) — системный промпт + вопрос + ожидаемый ответ
└── budget            ← сюда влезают чанки
```

### Инициализация

```python
manager = TokenBudgetManager(
    model="gpt-4o",
    max_context_tokens=None,   # None → берётся из таблицы по модели
    reserved_tokens=1_000,
)
```

Таблица известных окон:
- `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` → 128 000
- `claude-*` → 200 000
- Неизвестные модели → 8 192 (консервативный запасной вариант)

Токенизация через `tiktoken` — тот же BPE, что и у целевой модели. Для Ollama-моделей используется `cl100k_base`.

### `fit_chunks(chunks)` — обрезка по бюджету

```python
selected = manager.fit_chunks(chunks)
```

Чанки уже отсортированы по релевантности (лучшие первые). Берём их по порядку, пока не кончится бюджет:

```
чанк 1 (300 токенов) → влезает, берём
чанк 2 (400 токенов) → влезает, берём
чанк 3 (500 токенов) → не влезает, СТОП (не пропускаем и не берём следующий)
```

Важно: при переполнении бюджета итерация **прекращается**, а не пропускает чанк. Иначе мы бы взяли менее релевантные чанки вместо более релевантного.

### `remaining(chunks)` — остаток бюджета

```python
left = manager.remaining(chunks)   # сколько токенов осталось после размещения чанков
```

Полезно для диагностики: если остаток большой — чанки умещаются, если = 0 — контекст заполнен под завязку.
