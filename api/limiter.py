"""Общий экземпляр slowapi Limiter.

Импортируется в app.py (регистрация) и в роутеры (декораторы).
key_func=get_remote_address — лимит считается по IP клиента.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
