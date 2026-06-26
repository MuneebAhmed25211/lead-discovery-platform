"""
routers/discovery.py
─────────────────────
POST /api/v1/discovery/search

Thin HTTP layer over DiscoveryService. Validates the request, resolves
dependencies, calls the service, and wraps the result in the standard
SuccessResponse envelope.

This endpoint is intentionally NOT gated by the fingerprint rate-limit
middleware — quota is consumed once at /icp/generate, and discovery is
the natural next step within the same already-paid-for session. Gating
it again would mean one logical "search" costs two quota units.

A fingerprint IS still required (via require_fingerprint) for analytics
attribution, even though it doesn't trigger a rate-limit check here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from supabase import Client

from cache.supabase_cache import get_cache
from core.dependencies import get_db, require_fingerprint
from core.logging import get_logger
from schemas.common import SuccessResponse
from schemas.discovery import DiscoveryRequest, DiscoveryResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/discovery/search",
    response_model=SuccessResponse[DiscoveryResponse],
    tags=["Pipeline"],
)
async def discover_companies(
    body: DiscoveryRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[DiscoveryResponse]:
    """
    Discover company candidates matching an ICP's search queries.

    Runs each ICP search query through Serper, falling back to Tavily on
    failure or zero results. Normalizes and deduplicates by root domain,
    persists candidates to `discovered_companies`, and returns them.

    Always returns 200 with a DiscoveryResponse — even when all providers
    fail, in which case `companies` is empty and `partial_failure` is true.
    The only error response is 404 if `icp_id` does not exist.
    """
    from services.discovery_service import DiscoveryService

    cache = get_cache(db)
    service = DiscoveryService(db=db, cache=cache)

    result = await service.discover(search_id=body.search_id, icp_id=body.icp_id)

    _log_analytics(
        db,
        fingerprint=fingerprint,
        search_id=body.search_id,
        icp_id=body.icp_id,
        total_found=result.total_found,
        queries_failed=result.queries_failed,
        partial_failure=result.partial_failure,
    )

    logger.info(
        "discovery_request_complete",
        search_id=body.search_id,
        total_found=result.total_found,
        partial_failure=result.partial_failure,
    )

    return SuccessResponse(data=result, meta={"search_id": body.search_id})


def _log_analytics(
    db: Client,
    fingerprint: str,
    search_id: str,
    icp_id: str,
    total_found: int,
    queries_failed: int,
    partial_failure: bool,
) -> None:
    try:
        db.table("analytics_events").insert({
            "event_type": "discovery_complete",
            "fingerprint": fingerprint[:8],
            "metadata": {
                "search_id": search_id,
                "icp_id": icp_id,
                "total_found": total_found,
                "queries_failed": queries_failed,
                "partial_failure": partial_failure,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as exc:
        logger.debug("analytics_log_failed", error=str(exc))
