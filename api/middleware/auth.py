"""Аутентификация: X-API-Key и JWT Bearer.

Оба метода реализованы как одна FastAPI-зависимость ``require_auth``,
которую можно подключить глобально (в app.py) или к отдельным роутерам.

Схема проверки:
    1. Если заголовок ``X-API-Key`` совпадает с ``settings.api_secret_key`` → OK.
    2. Если заголовок ``Authorization: Bearer <token>`` содержит валидный JWT
       подписанный тем же ``api_secret_key`` (алгоритм из JWT_ALGORITHM) → OK.
    3. Иначе → 401 Unauthorized.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from config import settings

# FastAPI-дескрипторы для извлечения заголовков.
# auto_error=False — не бросать исключение самостоятельно, мы обработаем ниже.
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_BEARER = HTTPBearer(auto_error=False)


async def require_auth(
    api_key: str | None = Security(_API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> None:
    """FastAPI-зависимость: проверяет X-API-Key или JWT Bearer токен.

    Подключается через ``Depends(require_auth)`` или как глобальная зависимость
    приложения через ``dependencies=[Depends(require_auth)]``.

    Raises:
        HTTPException 401: если ни один из методов аутентификации не прошёл.
    """
    # --- X-API-Key ---
    if api_key and api_key == settings.api_secret_key:
        return

    # --- JWT Bearer ---
    if bearer:
        try:
            # decode проверяет подпись и срок действия (exp) автоматически.
            jwt.decode(bearer.credentials, settings.api_secret_key, algorithms=[settings.jwt_algorithm])
            return
        except JWTError:
            # Невалидная подпись, истёкший токен или искажённый payload.
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверные или отсутствующие учётные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
