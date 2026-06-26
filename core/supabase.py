"""
core/supabase.py
────────────────
Supabase client management.

We maintain TWO clients:
  - anon_client: uses the anon key — for operations that mirror what the
    frontend could do. Respects Row Level Security (RLS) policies.
  - service_client: uses the service role key — bypasses RLS. Used only
    for server-side operations like writing rate limit records, cache
    entries, and analytics events. Never expose to the browser.

Both clients are created once at startup (via lifespan) and reused for
the lifetime of the process. The Supabase Python client is thread-safe
and handles connection pooling internally.

Usage:
    from core.supabase import get_service_client

    async def example():
        db = get_service_client()
        result = db.table("companies").select("*").eq("id", company_id).execute()
"""

from __future__ import annotations

from supabase import Client, create_client

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Module-level singletons — populated by init_supabase()
_anon_client: Client | None = None
_service_client: Client | None = None


def init_supabase() -> None:
    """
    Create and cache both Supabase clients.
    Call once from the FastAPI lifespan startup handler.
    """
    global _anon_client, _service_client

    url = str(settings.SUPABASE_URL)

    logger.info("supabase_init", url=url)

    _anon_client = create_client(url, settings.SUPABASE_ANON_KEY)
    _service_client = create_client(url, settings.SUPABASE_SERVICE_KEY)

    logger.info("supabase_ready")


def get_anon_client() -> Client:
    """Return the anon (RLS-respecting) Supabase client."""
    if _anon_client is None:
        raise RuntimeError("Supabase not initialised. Call init_supabase() first.")
    return _anon_client


def get_service_client() -> Client:
    """
    Return the service-role Supabase client.

    WARNING: This client bypasses Row Level Security.
    Use only for server-side writes (rate limits, cache, analytics).
    Never pass this client to any code path that processes user input
    without additional validation.
    """
    if _service_client is None:
        raise RuntimeError("Supabase not initialised. Call init_supabase() first.")
    return _service_client


def health_check() -> dict[str, str]:
    """
    Verify both clients can reach Supabase.
    Used by the /health endpoint.
    Returns {"status": "ok"} or {"status": "error", "detail": "..."}.
    """
    try:
        client = get_service_client()
        # A lightweight query — just fetch the server time
        client.rpc("now").execute()
        return {"status": "ok"}
    except Exception as exc:
        logger.warning("supabase_health_check_failed", error=str(exc))
        return {"status": "error", "detail": str(exc)}
