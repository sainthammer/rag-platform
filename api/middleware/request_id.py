"""Middleware: генерирует X-Request-ID для каждого запроса.

Поведение:
- Если клиент прислал заголовок ``X-Request-ID`` — используем его (идемпотентность
  при повторных запросах и для сквозной трассировки через несколько сервисов).
- Иначе генерируем UUID4.
- ID сохраняется в ``request.state.request_id`` (доступен в роутерах).
- Прокидывается в активный OTel-спан как атрибут ``http.request_id``.
- Возвращается клиенту в заголовке ответа ``X-Request-ID``.
"""

import uuid

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        request.state.request_id = request_id

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("http.request_id", request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
