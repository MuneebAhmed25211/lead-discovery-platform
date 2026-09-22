"""
routers/icp.py
──────────────
POST /api/v1/icp/generate

Endpoint 1 of the pipeline. Accepts a service description, returns a
structured ICP. This is the ONLY endpoint that consumes one Groq call
upfront and that the rate limiter gates on.

Flow:
  1. Validate request (Pydantic)
  2. Rate limit check happens in middleware BEFORE this handler runs
  3. Check cache — return cached ICP if hit (no Groq call)
  4. Call ICP service (Groq call #1)
  5. Store ICP in Supabase
  6. Increment rate limit counter (only after success)
  7. Log analytics event
  8. Return ICP

The handler is kept thin — all logic lives in icp_service.py.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from supabase import Client

from cache.supabase_cache import SupabaseCache, get_cache
from core.config import settings
from core.dependencies import get_client_ip, get_db, require_fingerprint
from core.errors import DatabaseError
from core.logging import get_logger
from middleware.rate_limit import RateLimitService
from schemas.common import SuccessResponse
from schemas.icp import ICPGenerateRequest, ICPGenerateResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post("/icp/generate", response_model=SuccessResponse[ICPGenerateResponse], tags=["Pipeline"])
async def generate_icp(
    body: ICPGenerateRequest,
    request: Request,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[ICPGenerateResponse]:
    """
    Generate an Ideal Customer Profile from a service description.

    Rate limited: 3 free calls per 30-day window per fingerprint.
    Cached: same description returns cached ICP for 24 hours.
    """
    # Lazy import to avoid circular at module load
    from services.icp_service import ICPService

    cache = get_cache(db)
    service = ICPService(db=db, cache=cache)

    # Always generate search_id server-side. If the client provides one,
    # accept it only if it looks like a valid identifier (not a description
    # string). This closes the bug where the service_description was
    # accidentally used as search_id by misconfigured frontends.
    provided = body.search_id
    if provided and len(provided) >= 8 and len(provided) <= 128 and " " not in provided:
        search_id = provided
    else:
        search_id = str(uuid.uuid4())

    # Store search record
    _store_search_record(db, search_id=search_id, fingerprint=fingerprint, description=body.service_description)

    # Generate (or retrieve cached) ICP
    result = await service.generate(
        service_description=body.service_description,
        search_id=search_id,
    )

    # Increment rate limit ONLY on success (not on cache hit to be fair)
    if not result.cached:
        ip = get_client_ip(request)
        rl_service = RateLimitService(db)
        await rl_service.increment(fingerprint=fingerprint, ip=ip)

    # Log analytics event
    _log_analytics(db, event_type="icp_generated", fingerprint=fingerprint, metadata={
        "search_id": search_id,
        "cached": result.cached,
    })

    logger.info("icp_generated", search_id=search_id, cached=result.cached)

    return SuccessResponse(data=result, meta={"search_id": search_id})


def _store_search_record(db: Client, search_id: str, fingerprint: str, description: str) -> None:
    """
    Persist the searches row that icps.search_id has a foreign-key
    dependency on.

    BUG FIX: this used to catch every exception and only log a warning,
    letting generate_icp() carry on to the Groq call and the ICP insert
    even when this row was never created. Confirmed real case: a
    transient network failure reaching Supabase here (getaddrinfo
    failed — a DNS blip) was silently swallowed; generate_icp() then
    spent 10-20+ seconds on a real Groq call, and only THEN failed with
    a foreign-key violation on the icps insert ("Key (search_id)=(...)
    is not present in table searches") — which looks like an ICP-storage
    bug but is actually caused entirely by this earlier silent failure.
    Raising here means a transient DB/network issue fails fast and
    cheaply, before wasting a Groq call, with the error pointing at its
    actual source.
    """
    try:
        db.table("searches").upsert({
            "id": search_id,
            "fingerprint": fingerprint,
            "service_description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="id").execute()
    except Exception as exc:
        logger.error("search_record_store_failed", error=str(exc))
        raise DatabaseError(operation="upsert", table="searches", cause=str(exc)) from exc


def _log_analytics(db: Client, event_type: str, fingerprint: str, metadata: dict) -> None:
    try:
        db.table("analytics_events").insert({
            "event_type": event_type,
            "fingerprint": fingerprint[:8],
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as exc:
        logger.debug("analytics_log_failed", error=str(exc))