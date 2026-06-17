"""
Инструменты ???
"""

import uuid


def to_uuid(s: str) -> str:
    """
    Функция перевода строки в uuid(в проектк используется для унификации id документов)
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def reciprocal_rank_fusion(result_lists, k: int = 60):
    """
    Объединяет несколько ранжированных списков id в один через RRF.

    args:
        result_lists: список списков id (каждый — отдельное ранжирование)
        k: сглаживающая константа (стандартно 60)
    """
    scores = {}
    for ids in result_lists:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
