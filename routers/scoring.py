"""
routers/scoring.py
────────────────────
POST /api/v1/scoring/score  — score a single researched company against an ICP
POST /api/v1/scoring/batch  — score every complete company for a search

Zero Groq calls — pure deterministic computation (see services/scoring_service.py)
so there is no caching layer here, unlike icp/research/qualify. Recomputing a
score is cheap and keeps it fresh if evidence has changed; score_batch already
skips companies that were already scored (see ScoringService._already_scored).

Both endpoints require X-Fingerprint, matching every other pipeline route.
Neither consumes rate-limit quota — that's spent once at ICP generation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from supabase import Client

from core.dependencies import get_db, require_fingerprint
from core.logging import get_logger
from schemas.common import SuccessResponse
from schemas.score import ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post("/scoring/score", response_model=SuccessResponse[ScoreResponse], tags=["Pipeline"])
async def score_company(
    body: ScoreRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[ScoreResponse]:
    """
    Score a single researched company against an ICP.

    Deterministic — evaluates the company's Phase 4 evidence across six
    weighted dimensions plus a disqualifier deduction, each with its own
    traceable reasoning and evidence_ids. Always recomputes and upserts
    (on_conflict=company_id), so re-scoring after new evidence is safe.
    """
    from services.scoring_service import ScoringService

    service = ScoringService(db=db)
    score = service.score_company(company_id=body.company_id, icp_id=body.icp_id)

    _log_analytics(db, fingerprint, "company_scored", {
        "company_id": body.company_id,
        "total_score": score.total_score,
        "score_category": score.score_category,
    })

    return SuccessResponse(data=ScoreResponse(score=score, cached=False))


@router.post("/scoring/batch", response_model=SuccessResponse[ScoreBatchResponse], tags=["Pipeline"])
async def score_batch(
    body: ScoreBatchRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[ScoreBatchResponse]:
    """
    Score every company with research_status='complete' for a search_id.

    Skips companies already scored and companies below min_research_confidence.
    Returns scores sorted highest-first.
    """
    from services.scoring_service import ScoringService

    service = ScoringService(db=db)
    result = service.score_batch(
        search_id=body.search_id,
        icp_id=body.icp_id,
        min_research_confidence=body.min_research_confidence,
    )

    _log_analytics(db, fingerprint, "batch_scoring_complete", {
        "search_id": body.search_id,
        "scored": result.scored,
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