"""
core/dependencies.py
────────────────────
FastAPI dependency injection providers.

These functions are used with FastAPI's Depends() mechanism.
They provide typed, validated objects to route handlers without
the handlers needing to know how those objects are constructed.

Pattern:
    @router.post("/endpoint")
    async def handler(
        db: ServiceClient = Depends(get_db),
        fp: str = Depends(require_fingerprint),
    ):
        ...
"""

from __future__ import annotations

from fastapi import Depends, Header, Request
from supabase import Client

from core.config import Settings, settings
from core.errors import FingerprintMissingError
from core.supabase import get_service_client


# ── Database ──────────────────────────────────────────────────────────────────

def get_db() -> Client:
    """
    Inject the Supabase service client.

    This is the primary DB dependency for all server-side operations.
    The client is a module-level singleton so FastAPI does not create
    a new connection on every request.
    """
    return get_service_client()


# Convenience type alias for route annotations
ServiceClient = Client


# ── Settings ──────────────────────────────────────────────────────────────────

def get_settings() -> Settings:
    """
    Inject the application settings.
    Useful in tests where you want to override individual settings.
    """
    return settings


# ── Fingerprint ───────────────────────────────────────────────────────────────

async def require_fingerprint(
    x_fingerprint: str | None = Header(default=None, alias="X-Fingerprint"),
) -> str:
    """
    Extract and validate the X-Fingerprint header.

    FingerprintJS Pro visitor IDs are 32-character hex strings.
    We accept any non-empty string >= 8 chars to be compatible with
    different fingerprinting libraries during development.

    Raises FingerprintMissingError (→ 400) if absent or too short.
    """
    if not x_fingerprint or len(x_fingerprint.strip()) < 8:
        raise FingerprintMissingError()
    return x_fingerprint.strip()


async def optional_fingerprint(
    x_fingerprint: str | None = Header(default=None, alias="X-Fingerprint"),
) -> str | None:
    """
    Like require_fingerprint but returns None instead of raising.
    Used for endpoints that degrade gracefully without a fingerprint
    (e.g. health check, public landing page data).
    """
    if not x_fingerprint or len(x_fingerprint.strip()) < 8:
        return None
    return x_fingerprint.strip()


# ── Client IP ─────────────────────────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """
    Extract the real client IP, respecting Vercel/Render proxy headers.

    Priority:
      1. X-Forwarded-For (set by Vercel edge, Render proxy)
      2. X-Real-IP
      3. request.client.host (direct connection)
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list; take the first (real client)
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    if request.client:
        return request.client.host

    return "unknown"
