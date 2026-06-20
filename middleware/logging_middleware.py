"""
middleware/logging_middleware.py
─────────────────────────────────
Request/response access logging middleware.

Logs every request with:
  - method, path, status_code
  - duration_ms
  - client IP
  - fingerprint (first 8 chars only, for privacy)

In production this feeds into the log drain for monitoring.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

import structlog

from core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Bind request_id to structlog context so all logs within
        # this request share it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        fp = request.headers.get("X-Fingerprint", "")[:8]

        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            ip=request.client.host if request.client else "unknown",
            fp=fp or "none",
        )

        response.headers["X-Request-ID"] = request_id
        return response
