"""
Инструменты ???
"""

import uuid


def to_uuid(s: str) -> str:
    """
    Функция перевода строки в uuid(в проектк используется для унификации id документов)
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))
