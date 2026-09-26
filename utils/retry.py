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

import asyncio
import functools
import time
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

# ── Global Groq call pacing ───────────────────────────────────────────────────
#
# Added after confirming in production logs that a single CSV-import batch of
# ~15 companies firing qualify calls back-to-back hit Groq's free-tier
# token-per-minute ceiling directly: "Limit 8000, Used 7536, Requested 3956" —
# a single qualify call runs ~3,400-4,000 tokens, so the free tier only has
# genuine headroom for roughly 2 such calls per rolling minute. No amount of
# client-side pacing can stretch a fixed per-minute token budget further than
# that; what pacing CAN do is stop this app from being the one bursting past
# it, whether that burst comes from one batch or from several different
# visitors' requests landing close together (this app runs as a single Render
# process — `--workers 1` — so a process-wide gate here genuinely covers every
# concurrent request, not just calls within one batch).
#
# _MIN_GROQ_CALL_INTERVAL_SECONDS is deliberately a single named constant to
# make the tradeoff explicit and easy to tune later: lower it for faster
# batches with a higher chance of hitting a 429 (which @groq_retry above still
# absorbs via backoff — this isn't the only safety net, just the first one);
# raise it for a stronger guarantee at the cost of slower processing. 20s
# keeps a 15-company batch to ~5 minutes rather than firing all 15 in a few
# seconds, while meaningfully — not perfectly — reducing collisions, since
# true zero-collision safety would require ~30s (8000 TPM / ~4000 tokens per
# call ≈ 2 calls/minute) and that felt too slow for a live demo to accept by
# default without seeing real traffic patterns first.
_MIN_GROQ_CALL_INTERVAL_SECONDS = 20.0

_groq_pacing_lock = asyncio.Lock()
_last_groq_call_at: float = 0.0


async def _wait_for_groq_pacing_slot() -> None:
    """
    Block the caller until at least _MIN_GROQ_CALL_INTERVAL_SECONDS have
    elapsed since the last Groq call *started*, across the whole process —
    not just within one request or one batch. The lock makes the
    read-compute-update sequence atomic, so two concurrent calls can't both
    read the same "last call" timestamp and both proceed as if they were
    first; only one waiter can be inside the critical section at a time,
    and each one updates the shared timestamp before releasing it for the
    next.
    """
    global _last_groq_call_at
    async with _groq_pacing_lock:
        now = time.monotonic()
        elapsed = now - _last_groq_call_at
        remaining = _MIN_GROQ_CALL_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            logger.info("groq_pacing_wait", seconds=round(remaining, 1))
            await asyncio.sleep(remaining)
        _last_groq_call_at = time.monotonic()


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

        # BUG FIX: this used to have the GroqError/GroqRateLimitError
        # fallback logic written as plain code AFTER this async-for loop,
        # on the assumption that once stop_after_attempt(3) is reached the
        # loop simply ends and falls through. It doesn't: with
        # reraise=False, tenacity's AttemptManager.__exit__ raises its own
        # tenacity.RetryError the moment the stop condition is hit — that
        # exception escapes the "async for" statement itself, so the code
        # written after the loop was unreachable dead code. Confirmed by
        # reproducing this exact pattern standalone: the intended fallback
        # line never executes, and a raw, un-typed tenacity.RetryError
        # propagates instead. Since RetryError isn't a LeadDiscoveryError
        # subclass, main.py's global handler doesn't recognize it either,
        # so it surfaced as an unhandled exception / raw 500 with a full
        # Python traceback dumped to the console — confirmed in live
        # testing (ICP generation crashed exactly this way under a
        # transient network blip). Wrapping the loop in try/except and
        # doing the classification in the except block (which DOES run,
        # since the escaping RetryError is catchable from outside the
        # loop) fixes this while keeping the retry behavior identical.
        try:
            async for attempt_ctx in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=8),
                reraise=False,
            ):
                with attempt_ctx:
                    attempt += 1
                    try:
                        await _wait_for_groq_pacing_slot()
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        logger.warning("groq_retry", operation=func.__name__, attempt=attempt, error=str(exc))
                        raise
        except Exception as exc:
            operation = func.__name__
            cause = str(last_exc) if last_exc is not None else str(exc)
            error_msg = cause.lower()
            if "rate_limit" in error_msg or "429" in error_msg:
                logger.warning("groq_rate_limit", operation=operation, attempt=attempt)
                raise GroqRateLimitError(operation=operation) from exc
            logger.error("groq_failed", operation=operation, attempts=attempt, cause=cause)
            raise GroqError(operation=operation, cause=cause) from exc

        # Unreachable in practice (the try/except above always raises or
        # returns first) but kept as a defensive fallback in case tenacity's
        # exhaustion behavior ever changes.
        raise GroqError(operation=func.__name__, cause=str(last_exc) if last_exc else "unknown")

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

        # Same reraise=False escaping-RetryError issue as groq_retry above —
        # see that function's comment for the full explanation. This
        # decorator has no current callers (grep confirms nothing in the
        # codebase uses @http_retry today), so it had no live impact, but
        # fixed for consistency and so it's safe to reach for later.
        try:
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
        except CrawlTimeoutError:
            raise
        except Exception:
            logger.warning("http_failed", fn=func.__name__, attempts=attempt, cause=str(last_exc))
            return None

        return None  # defensive fallback, same reasoning as groq_retry

    return wrapper  # type: ignore[return-value]