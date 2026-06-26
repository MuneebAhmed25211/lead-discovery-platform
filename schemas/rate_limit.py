"""
schemas/rate_limit.py
─────────────────────
Schemas for rate limit state and responses.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from schemas.common import AppBaseModel, UUIDModel


class RateLimitState(AppBaseModel):
    """Current state of a fingerprint's rate limit window."""

    fingerprint: str
    ip_address: str
    search_count: int
    max_searches: int
    window_days: int
    first_search_at: datetime | None = None
    last_search_at: datetime | None = None
    blocked_until: datetime | None = None

    @property
    def searches_remaining(self) -> int:
        return max(0, self.max_searches - self.search_count)

    @property
    def is_blocked(self) -> bool:
        if self.search_count < self.max_searches:
            return False
        if self.blocked_until is None:
            return True
        return datetime.utcnow() < self.blocked_until.replace(tzinfo=None)


class RateLimitRecord(UUIDModel):
    """Supabase `rate_limits` table row."""

    fingerprint: str
    ip_address: str
    search_count: int
    first_search_at: datetime
    last_search_at: datetime
    blocked_until: datetime | None


class RateLimitStatusResponse(AppBaseModel):
    """Response from GET /api/v1/rate-limit/status"""

    searches_used: int
    searches_allowed: int
    searches_remaining: int
    is_blocked: bool
    window_resets_at: datetime | None = None
    contact_url: str = ""
