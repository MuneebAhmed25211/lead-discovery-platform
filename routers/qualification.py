"""
routers/qualification.py
────────────────────────
POST /api/v1/qualify/company  — qualify a single company against an ICP (Groq call #2)
POST /api/v1/qualify/batch    — qualify every scored company above a threshold

Both require X-Fingerprint. Neither consumes rate-limit quota (spent once
at ICP generation, same as research/scoring). Qualification responses are
cached exactly like ICP generation — see cache/supabase_cache.py and
CACHE_QUALIFY_TTL_HOURS in core/config.py — so re-qualifying the same
(company, icp) pair does not repeat the Groq call.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from supabase import Client

from cache.supabase_cache import get_cache
from core.dependencies import get_db, require_fingerprint
from core.logging import get_logger
from schemas.common import SuccessResponse
from schemas.qualification import (
    QualifyBatchRequest,
    QualifyBatchResponse,
    QualifyRequest,
    QualifyResponse,
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("/qualify/company", response_model=SuccessResponse[QualifyResponse], tags=["Pipeline"])
async def qualify_company(
    body: QualifyRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[QualifyResponse]:
    """
    Qualify a single researched company against an ICP.

    Groq call #2 — reasons only from the company's Phase 4 evidence, the
    ICP, and (if present) its deterministic lead score. Cached per
    (company_id, icp_id) pair for CACHE_QUALIFY_TTL_HOURS to avoid
    duplicate AI calls.
    """
    from services.qualify_service import QualifyService

    cache = get_cache(db)
    service = QualifyService(db=db, cache=cache)
    qualification, cached = await service.qualify_company(
        company_id=body.company_id, icp_id=body.icp_id,
    )

    _log_analytics(db, fingerprint, "company_qualified", {
        "company_id": body.company_id,
        "qualification_status": qualification.qualification_status,
        "confidence": qualification.confidence,
        "cached": cached,
    })

    return SuccessResponse(data=QualifyResponse(qualification=qualification, cached=cached))


@router.post("/qualify/batch", response_model=SuccessResponse[QualifyBatchResponse], tags=["Pipeline"])
async def qualify_batch(
    body: QualifyBatchRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[QualifyBatchResponse]:
    """
    Qualify every company for a search whose lead_score total_score >= min_score.

    Bounded concurrency (3 at a time, same cap as research_batch) to stay
    within Groq rate limits. Each qualification is independently cached,
    so re-running a batch after partial failure only re-calls Groq for
    companies that weren't already qualified.
    """
    from services.qualify_service import QualifyService

    cache = get_cache(db)
    service = QualifyService(db=db, cache=cache)
    result = await service.qualify_batch(
        search_id=body.search_id, icp_id=body.icp_id, min_score=body.min_score,
    )

    _log_analytics(db, fingerprint, "batch_qualification_complete", {
        "search_id": body.search_id,
        "qualified": result.qualified,
        "skipped": result.skipped,
        "total": result.total,
    })

    return SuccessResponse(data=result)


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