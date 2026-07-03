"""
routers/research.py
────────────────────
POST /api/v1/research/company  — research a single discovered company
POST /api/v1/research/batch    — research all unpromoted companies for a search

Both endpoints require X-Fingerprint. Neither is rate-limited — they
consume quota already spent at ICP generation.

The single endpoint is useful for testing one company during development.
The batch endpoint is what the frontend calls after discovery completes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from supabase import Client

from core.dependencies import get_db, require_fingerprint
from core.logging import get_logger
from schemas.common import SuccessResponse
from schemas.company import ResearchBatchRequest, ResearchRequest, ResearchResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post("/research/company", response_model=SuccessResponse[ResearchResponse], tags=["Pipeline"])
async def research_company(
    body: ResearchRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[ResearchResponse]:
    """
    Research a single discovered company.

    Crawls the company's website, runs all 9 deterministic extractors,
    persists evidence rows, creates a companies row, and marks the
    discovered_companies row as promoted.
    """
    from services.research_service import ResearchService

    service = ResearchService(db=db)
    result = await service.research_company(
        discovered_company_id=body.discovered_company_id,
    )

    _log_analytics(db, fingerprint, "company_researched", {
        "domain": result.company.domain,
        "evidence_count": result.evidence_count,
        "confidence": result.data_confidence,
        "status": result.company.research_status,
    })

    return SuccessResponse(data=result)


@router.post("/research/batch", tags=["Pipeline"])
async def research_batch(
    body: ResearchBatchRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[dict]:
    """
    Research all unpromoted discovered companies for a search_id.

    Processes up to max_companies (default 20) concurrently with a
    concurrency cap of 3. Returns a summary of results.
    """
    from services.research_service import ResearchService

    service = ResearchService(db=db)
    summary = await service.research_batch(
        search_id=body.search_id,
        max_companies=body.max_companies,
    )

    _log_analytics(db, fingerprint, "batch_research_complete", {
        "search_id": body.search_id,
        **summary,
    })

    return SuccessResponse(data=summary)


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
