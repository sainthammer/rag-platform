# Модуль 8 — evaluation/

**Файлы:** `evaluation/ragas_eval.py`, `evaluation/eval_runner.py`, `evaluation/testcase.py`, `evaluation/testcases_dataset.py`

Оценка качества RAG-пайплайна через фреймворк RAGAS.

## Зачем нужна оценка

RAG-система может быть «умной» в одном аспекте и проваливаться в другом:
- Находит правильные чанки, но LLM выдумывает (Faithfulness = 0)
- LLM отвечает хорошо, но поиск возвращает нерелевантные фрагменты (ContextRecall = 0)

Без оценки невозможно понять, где именно ломается система.

## Структура

```
evaluation/
  testcase.py           TestCase — датакласс одного теста
  testcases_dataset.py  набор из 46 тест-кейсов
  ragas_eval.py         RAGAS метрики, сборка датасета
  eval_runner.py        запуск оценки, HTML-отчёт
```

## testcase.py — TestCase

```python
@dataclass
class TestCase:
    question: str       # вопрос пользователя
    ground_truth: str   # эталонный правильный ответ
    source_doc: str     # из какого документа должен быть ответ
    category: str       # тип вопроса (factual, reasoning, comparison...)
```

Тест-кейс — это минимальная единица оценки: задаём вопрос системе, сравниваем ответ с эталоном.

## testcases_dataset.py — 46 тест-кейсов

Готовый датасет вопросов и эталонных ответов. Используется для запуска автоматической оценки.

## ragas_eval.py — RAGAS метрики

RAGAS (RAG Assessment) — специализированный фреймворк для оценки RAG-систем. Использует **LLM-as-judge**: сам использует языковую модель для оценки.

### Четыре метрики

| Метрика | Что измеряет | Вопрос |
|---|---|---|
| `Faithfulness` | Верность ответа контексту | Не выдумывает ли модель то, чего нет в чанках? |
| `ResponseRelevancy` | Релевантность ответа вопросу | Отвечает ли на то, о чём спросили? |
| `ContextPrecision` | Точность retrieval | Среди найденных чанков — сколько реально полезных? |
| `ContextRecall` | Полнота retrieval | Нашли ли мы все чанки, нужные для правильного ответа? |

### Как данные попадают в RAGAS

```python
rows = build_rows_from_testcases(
    testcases,
    answer_by_question={"вопрос": "ответ модели"},
    contexts_by_question={"вопрос": ["чанк 1", "чанк 2"]},
)
dataset = as_hf_dataset(rows)
```

Каждая строка датасета:
```python
{
    "question":     "Как настроить авторизацию?",
    "answer":       "Ответ, который дала модель",
    "contexts":     ["найденный чанк 1", "найденный чанк 2"],
    "ground_truth": "Правильный ответ из тест-кейса",
}
```

### Запуск оценки

```python
result = evaluate_ragas(
    dataset=dataset,
    llm=None,        # None → OpenAI (нужен OPENAI_API_KEY)
    embeddings=None, # None → OpenAI embeddings
)
```

Для локальной оценки без OpenAI — передать Ollama-обёртку:
```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

ollama_llm = LangchainLLMWrapper(ChatOpenAI(base_url="http://localhost:11434/v1", ...))
result = evaluate_ragas(dataset, llm=ollama_llm)
```

### Совместимость с ragas 0.4.x

В `ragas_eval.py` есть патч совместимости:
```python
# ragas 0.4.x делает import ChatVertexAI из пакета, который был удалён
# Инжектируем заглушку, чтобы import не падал
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub
```

## eval_runner.py — оркестратор оценки

### Два режима запуска

**`mode="mock"`** — быстрая проверка без LLM:
- Ответы генерируются фиктивно (заглушки)
- Полезно для проверки инфраструктуры оценки
- Не требует API-ключей

**`mode="ollama"`** — реальная оценка через Ollama:
- Пайплайн прогоняет реальные запросы через RAG
- RAGAS оценивает через Ollama LLM
- Нужен запущенный Ollama-сервер

### HTML-отчёт

После оценки генерируется `eval_report.html` с:
- Итоговыми RAGAS метриками (средними значениями)
- Таблицей по каждому тест-кейсу
- Информацией о прохождении проверки на галлюцинации

### API-интерфейс

```
POST /v1/eval  {"mode": "mock", "max_cases": 5}
  → 202 Accepted, {"job_id": "xyz"}    (запускает в фоне)

GET /v1/eval/xyz
  → {
      "status": "done",
      "report_path": "eval_report.html",
      "hallucination_pass": 4,
      "hallucination_total": 5,
      "ragas_avg": {
          "faithfulness": 0.87,
          "response_relevancy": 0.92,
          "context_precision": 0.78,
          "context_recall": 0.83
      }
    }
```

## Интерпретация метрик

| Значение | Интерпретация |
|---|---|
| 0.9–1.0 | Отлично |
| 0.7–0.9 | Хорошо, есть куда расти |
| 0.5–0.7 | Проблема, нужно разбираться |
| < 0.5 | Серьёзная проблема |

Низкий `Faithfulness` → LLM "галлюцинирует" (придумывает что не из контекста) → попробуй `STRICT` шаблон промпта.

Низкий `ContextRecall` → поиск не находит нужные документы → попробуй `HybridVectorStore` или CrossEncoderReranker.

Низкий `ContextPrecision` → поиск возвращает лишние нерелевантные чанки → уменьши `n_results` или повысь `score_threshold`.
