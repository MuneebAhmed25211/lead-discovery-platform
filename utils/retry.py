"""
utils/retry.py
──────────────
Reusable retry decorators built on Tenacity.

Two decorators are provided:

  @groq_retry   — for Groq API calls (3 attempts, exponential backoff,
                  detects rate-limit errors and raises GroqRateLimitError)

  @http_retry   — for outbound HTTP requests during crawling
                  (2 attempts, shorter backoff, raises CrawlError on exhaustion)

Both decorators log every retry attempt so we can monitor failure patterns.

Usage:
    from utils.retry import groq_retry, http_retry

    @groq_retry
    async def call_groq(...):
        ...

    @http_retry
    async def fetch_page(...):
        ...
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def groq_retry(func: F) -> F:
    """
    Retry decorator for Groq API calls.

    Strategy:
      - 3 total attempts (1 original + 2 retries)
      - Exponential backoff: 2s, 4s
      - Retries on any Exception except GroqRateLimitError
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        from core.errors import GroqError, GroqRateLimitError

        attempt = 0
        last_exc: Exception | None = None

        async for attempt_ctx in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            reraise=False,
        ):
            with attempt_ctx:
                attempt += 1
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    error_msg = str(exc).lower()
                    if "rate_limit" in error_msg or "429" in error_msg:
                        operation = func.__name__
                        logger.warning("groq_rate_limit", operation=operation, attempt=attempt)
                        raise GroqRateLimitError(operation=operation) from exc
                    last_exc = exc
                    logger.warning("groq_retry", operation=func.__name__, attempt=attempt, error=str(exc))
                    raise

        operation = func.__name__
        cause = str(last_exc) if last_exc else "unknown"
        logger.error("groq_failed", operation=operation, attempts=attempt, cause=cause)
        from core.errors import GroqError
        raise GroqError(operation=operation, cause=cause)

    return wrapper  # type: ignore[return-value]


def http_retry(func: F) -> F:
    """
    Retry decorator for outbound HTTP requests (crawl pages).

    Strategy:
      - 2 total attempts (1 original + 1 retry)
      - Does NOT retry on timeout (CrawlTimeoutError)
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        from core.errors import CrawlTimeoutError

        attempt = 0
        last_exc: Exception | None = None

        async for attempt_ctx in AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=2),
            reraise=False,
        ):
            with attempt_ctx:
                attempt += 1
                try:
                    return await func(*args, **kwargs)
                except CrawlTimeoutError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    logger.debug("http_retry", fn=func.__name__, attempt=attempt, error=str(exc))
                    raise

        logger.warning("http_failed", fn=func.__name__, attempts=attempt, cause=str(last_exc))
        return None

    return wrapper  # type: ignore[return-value]
