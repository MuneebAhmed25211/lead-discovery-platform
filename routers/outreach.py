"""
routers/outreach.py
────────────────────
POST /api/v1/outreach/generate — generate personalized outreach for a
qualified company (Groq call #3, the final Groq call in the pipeline).

Requires X-Fingerprint. Does not consume rate-limit quota (spent once
at ICP generation, same as research/scoring/qualify). Outreach responses
are cached exactly like qualification — see cache/supabase_cache.py and
CACHE_OUTREACH_TTL_HOURS in core/config.py — so regenerating outreach for
the same company does not repeat the Groq call.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from supabase import Client

from cache.supabase_cache import get_cache
from core.dependencies import get_db, require_fingerprint
from core.logging import get_logger
from schemas.common import SuccessResponse
from schemas.qualification import OutreachRequest, OutreachResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post("/outreach/generate", response_model=SuccessResponse[OutreachResponse], tags=["Pipeline"])
async def generate_outreach(
    body: OutreachRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[OutreachResponse]:
    """
    Generate personalized outreach messages for a single qualified company.

    Groq call #3 — the final Groq call in the pipeline. Writes from the
    company's Phase 4 evidence and its existing Phase 5 qualification
    only; performs no new fit analysis. Cached per company_id for
    CACHE_OUTREACH_TTL_HOURS to avoid duplicate AI calls.

    Requires the company to have already been qualified via
    POST /qualify/company — returns NOT_FOUND (404) otherwise.
    """
    from services.outreach_service import OutreachService

    cache = get_cache(db)
    service = OutreachService(db=db, cache=cache)
    outreach, cached = await service.generate_outreach(company_id=body.company_id)

    _log_analytics(db, fingerprint, "outreach_generated", {
        "company_id": body.company_id,
        "cached": cached,
    })

    return SuccessResponse(data=OutreachResponse(outreach=outreach, cached=cached))


def _log_analytics(db: Client, fingerprint: str, event: str, metadata: dict) -> None:
    try:
        db.table("analytics_events").insert({
            "event_type": event,
            "fingerprint": fingerprint[:8],
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as exc:
        logger.debug("analytics_log_failed", error=str(exc))