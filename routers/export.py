"""
routers/export.py
────────────────────
GET /api/v1/export/csv — download a search's companies (+ score,
qualification, and outreach data where available) as a CSV file.

Deterministic — no Groq calls, no caching layer (same reasoning as
scoring.py: rebuilding the CSV is cheap and always reflects the latest
data). Requires X-Fingerprint and only exports searches owned by that
fingerprint — this is a public multi-tenant platform, so one visitor
must never be able to pull another visitor's leads by guessing a
search_id. Does not consume rate-limit quota (spent once at ICP
generation, same as every other pipeline route).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from supabase import Client

from core.dependencies import get_db, require_fingerprint
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/export/csv", tags=["Pipeline"])
async def export_csv(
    search_id: str = Query(..., description="The search to export companies for."),
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> StreamingResponse:
    """
    Export every researched company for a search as CSV, one row per
    company, sorted by lead score (highest first). Columns for score,
    qualification, and outreach are left blank for any company that
    hasn't reached that pipeline stage yet — export works at any point
    in the pipeline, not only once everything is fully qualified.

    404s (via NotFoundError) if the search doesn't exist or doesn't
    belong to this fingerprint, rather than a 403 — never confirm to a
    caller that a search_id exists but belongs to someone else.
    """
    from services.export_service import ExportService

    service = ExportService(db=db)
    service.verify_ownership(search_id=search_id, fingerprint=fingerprint)
    csv_text = service.build_csv(search_id=search_id)

    _log_analytics(db, fingerprint, "csv_exported", {"search_id": search_id})

    filename = f"leads-{search_id}.csv"
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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