"""
middleware/rate_limit.py
────────────────────────
Rate limiting middleware and service.

Three-layer protection:
  Layer 1: localStorage (frontend) — instant UX feedback
  Layer 2: This middleware — server enforcement before route handler runs
  Layer 3: Supabase DB — authoritative, bypass-resistant

Only the ICP generate endpoint is rate-limited (one per pipeline run).
Subsequent steps (research, score, qualify, outreach) within the same
session are free — rate limiting at ICP is sufficient.
"""

from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from supabase import Client

from core.config import settings
from core.errors import RateLimitExceededError
from core.logging import get_logger
from core.supabase import get_service_client
from schemas.rate_limit import RateLimitState, RateLimitStatusResponse

logger = get_logger(__name__)

_RATE_LIMITED_PATHS = {"/api/v1/icp/generate"}


def resolve_quota(access_token: str | None, fingerprint: str) -> tuple[str, int]:
    """Return (quota_key, max_searches) for this request.

    A valid invite token moves the request onto its own quota bucket
    ("invite:<label>") with its own limit, shared across every browser/device
    the invitee uses. An invalid or missing token silently falls back to the
    normal per-fingerprint free quota (fail closed, never fail open).
    The fingerprint itself is untouched: search ownership still uses it.
    """
    token = (access_token or "").strip()
    if token:
        for entry in settings.RATE_LIMIT_INVITES.split(","):
            parts = entry.strip().split(":")
            if len(parts) != 3:
                continue
            label, secret, limit = parts
            if secret and hmac.compare_digest(token, secret):
                try:
                    return f"invite:{label}", int(limit)
                except ValueError:
                    break
    return fingerprint, settings.RATE_LIMIT_MAX_SEARCHES


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path not in _RATE_LIMITED_PATHS:
            return await call_next(request)

        fingerprint = request.headers.get("X-Fingerprint", "").strip()
        if len(fingerprint) < 8:
            return await call_next(request)

        client_ip = _get_client_ip(request)

        try:
            db = get_service_client()
            service = RateLimitService(db)
            token = request.headers.get("X-Access-Token")
            state = await service.check(fingerprint=fingerprint, ip=client_ip, access_token=token)

            if state.is_blocked:
                logger.info("rate_limit_blocked", fp=fingerprint[:8], count=state.search_count)
                error = RateLimitExceededError(
                    used=state.search_count,
                    max_allowed=state.max_searches,
                    contact_url=settings.RATE_LIMIT_CONTACT_URL,
                )
                return JSONResponse(
                    status_code=429,
                    content=error.to_dict(),
                    headers={
                        "X-RateLimit-Limit": str(state.max_searches),
                        "X-RateLimit-Remaining": "0",
                        "Retry-After": "2592000",
                    },
                )

            request.state.rate_limit = state

        except RateLimitExceededError:
            raise
        except Exception as exc:
            logger.error("rate_limit_check_failed", error=str(exc))

        return await call_next(request)


class RateLimitService:
    def __init__(self, db: Client) -> None:
        self._db = db

    async def check(self, fingerprint: str, ip: str, access_token: str | None = None) -> RateLimitState:
        fingerprint, max_searches = resolve_quota(access_token, fingerprint)
        window_days = settings.RATE_LIMIT_WINDOW_DAYS

        try:
            result = (
                self._db.table("rate_limits")
                .select("*")
                .eq("fingerprint", fingerprint)
                .single()
                .execute()
            )
            row = result.data
        except Exception:
            row = None

        if row is None:
            return RateLimitState(
                fingerprint=fingerprint, ip_address=ip,
                search_count=0, max_searches=max_searches, window_days=window_days,
            )

        first_search_at = _parse_dt(row["first_search_at"])
        window_end = first_search_at + timedelta(days=window_days)
        now = datetime.now(timezone.utc)

        if now > window_end:
            return RateLimitState(
                fingerprint=fingerprint, ip_address=ip,
                search_count=0, max_searches=max_searches, window_days=window_days,
            )

        blocked_until = _parse_dt(row["blocked_until"]) if row.get("blocked_until") else None

        return RateLimitState(
            fingerprint=fingerprint,
            ip_address=ip,
            search_count=row["search_count"],
            max_searches=max_searches,
            window_days=window_days,
            first_search_at=first_search_at,
            last_search_at=_parse_dt(row["last_search_at"]),
            blocked_until=blocked_until,
        )

    async def increment(self, fingerprint: str, ip: str, access_token: str | None = None) -> RateLimitState:
        """Increment search count. Call AFTER successful ICP generation."""
        fingerprint, max_searches = resolve_quota(access_token, fingerprint)
        window_days = settings.RATE_LIMIT_WINDOW_DAYS
        now = datetime.now(timezone.utc)

        try:
            result = (
                self._db.table("rate_limits")
                .select("search_count, first_search_at")
                .eq("fingerprint", fingerprint)
                .single()
                .execute()
            )
            row = result.data
        except Exception:
            row = None

        try:
            if row:
                first_search_at = _parse_dt(row["first_search_at"])
                window_end = first_search_at + timedelta(days=window_days)

                if now > window_end:
                    new_count = 1
                    self._db.table("rate_limits").update({
                        "search_count": 1,
                        "first_search_at": now.isoformat(),
                        "last_search_at": now.isoformat(),
                        "ip_address": ip,
                        "blocked_until": None,
                    }).eq("fingerprint", fingerprint).execute()
                else:
                    new_count = row["search_count"] + 1
                    update: dict = {
                        "search_count": new_count,
                        "last_search_at": now.isoformat(),
                        "ip_address": ip,
                    }
                    if new_count >= max_searches:
                        update["blocked_until"] = (now + timedelta(days=window_days)).isoformat()
                    self._db.table("rate_limits").update(update).eq("fingerprint", fingerprint).execute()
            else:
                new_count = 1
                self._db.table("rate_limits").insert({
                    "fingerprint": fingerprint,
                    "ip_address": ip,
                    "search_count": 1,
                    "first_search_at": now.isoformat(),
                    "last_search_at": now.isoformat(),
                }).execute()

            logger.info("rate_limit_incremented", fp=fingerprint[:8], count=new_count)
        except Exception as exc:
            logger.error("rate_limit_increment_failed", error=str(exc))

        return await self.check(fingerprint=fingerprint, ip=ip, access_token=access_token)

    async def get_status(self, fingerprint: str, ip: str, access_token: str | None = None) -> RateLimitStatusResponse:
        state = await self.check(fingerprint=fingerprint, ip=ip, access_token=access_token)
        window_resets_at = None
        if state.first_search_at:
            window_resets_at = state.first_search_at + timedelta(days=settings.RATE_LIMIT_WINDOW_DAYS)
        return RateLimitStatusResponse(
            searches_used=state.search_count,
            searches_allowed=state.max_searches,
            searches_remaining=state.searches_remaining,
            is_blocked=state.is_blocked,
            window_resets_at=window_resets_at,
            contact_url=settings.RATE_LIMIT_CONTACT_URL,
        )


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
