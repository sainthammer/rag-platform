# Модуль `evaluation/` — датасет тесткейсов и оценка через RAGAS

Модуль отвечает за подготовку тесткейсов и запуск автоматической оценки качества RAG.

Ключевая идея: у нас есть фиксированный набор вопросов (`TestCase`), а ответы/контексты
для этих вопросов поставляет уже ваша RAG-система. Далее эти данные конвертируются в
формат HuggingFace Dataset и оцениваются через RAGAS.

---

## Структура файлов

```
evaluation/
├── testcase.py          # dataclass TestCase
├── testcases_dataset.py # фиксированный набор тесткейсов (30/10/5)
├── ragas_eval.py        # утилиты подготовки датасета и запуск RAGAS
├── __init__.py          # re-export публичного API модуля
└── README.md
```

---

## Публичное API модуля

Удобный импорт:

```python
from evaluation import (
    TestCase,
    get_testcases,
    build_rows_from_testcases,
    as_hf_dataset,
    evaluate_ragas,
)
```

---

## `TestCase` — формат одного теста

`evaluation/testcase.py`:

- `question`: вопрос пользователя
- `ground_truth`: эталонный ответ (для negative может быть пустым)
- `source_doc`: ссылка на документ/источник в базе знаний
- `category`: категория (`positive`, `negative`, `multi_hop`)

---

## Датасет тесткейсов (30 positive / 10 negative / 5 multi-hop)

`evaluation/testcases_dataset.py` содержит `get_testcases()` — фиксированный список.

### Где хранится база знаний

Путь к базе знаний задаётся через переменную окружения:

```
KNOWLEDGE_BASE_PATH
```

- если не задана — используется значение по умолчанию `Phython_Theory`
- `source_doc` формируется как: `<KNOWLEDGE_BASE_PATH>/<relative_path>`

Это позволяет хранить базу знаний вне репозитория, не меняя код тесткейсов.

---

## Подготовка данных для RAGAS

RAGAS ожидает таблицу (Dataset) с полями:

- `question`
- `answer`
- `contexts` (list[str])
- `ground_truth`

Мы собираем это в 3 шага:

1) Берём тесткейсы: `testcases = get_testcases()`
2) Подставляем ответы/контексты вашей системы:

```python
rows = build_rows_from_testcases(
    testcases,
    answer_by_question=answers,
    contexts_by_question=contexts,
)
```

3) Конвертируем в HF Dataset:

```python
dataset = as_hf_dataset(rows)
```

---

## Запуск оценки RAGAS

`evaluation/ragas_eval.py` содержит `evaluate_ragas(dataset)`.

Используемые метрики:

- `faithfulness`
- `answer_relevancy`
- `context_recall`
- `context_precision`

Запуск:

```python
result = evaluate_ragas(dataset)
```

---

## Зависимости

RAGAS и HuggingFace Datasets оформлены как optional dependencies.
Установка:

```bash
pip install -e ".[eval]"
```

---

## Запуск `example.py`

### Режим OpenAI (требует API-ключ)

**1. Задать ключ в `.env`:**

```
OPENAI_API_KEY=sk-...
```

**2. Запустить:**

```powershell
# из корня проекта
python evaluation\example.py
```

### Режим Ollama (без API-ключа, локально)

**1. Установить и запустить Ollama:**

```powershell
# Скачать установщик: https://ollama.com/download/windows
# После установки в отдельном терминале:
ollama serve
```

**2. Скачать модели (один раз):**

```powershell
# LLM для оценки метрик faithfulness / context_precision / context_recall
ollama pull mistral:7b

# Embeddings для метрики response_relevancy
ollama pull nomic-embed-text

# Убедиться что модели доступны
ollama list
```

**3. Запустить example:**

```powershell
python evaluation\example.py ollama
```

> Оценка запускается на 5 mock-кейсах — реальные ответы и контексты
> подставляются вашим RAG-пайплайном через `build_rows_from_testcases()`.

### Примечания

- Метрики раcсчитываются по всем 5 кейсам и усредняются.
- `ResponseRelevancy` использует embeddings; остальные метрики — только LLM.
- При значениях `nan` в результатах проверьте, что модели скачаны и `ollama serve` запущен.
