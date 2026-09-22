"""
routers/import_csv.py
────────────────────────
POST /api/v1/import/parse    — parse an uploaded CSV, no cost incurred.
POST /api/v1/import/resolve  — resolve + validate the (capped) rows and
                                persist them, same contract as Discovery.

Two-step by design, matching the agreed UI flow: the frontend calls
/parse first and shows the person exactly what will happen —
    "Your file has {total_rows} companies. This demo processes the first
     {cap} — want your full list handled? Contact Muneeb Ahmed
     (muneeb.codeaxis@gmail.com / linkedin.com/in/muneebahmedai) for a
     custom build."
— BEFORE calling /resolve, which is the step that actually spends
search-provider (and, once the frontend feeds the results into the
existing Research/Score/Qualify flow, Groq) budget. /parse itself never
resolves a single row, so a person can upload, see the count, and decide
without cost.

Not gated by the fingerprint rate-limit middleware, for the same reason
discovery.router isn't: quota is consumed once at /icp/generate, and
whichever path (search-discovery or CSV-import) a person takes next is
the natural continuation of that same paid-for session, not a second
billable action.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile, File
from supabase import Client

from cache.supabase_cache import get_cache
from core.dependencies import get_db, require_fingerprint
from core.errors import InputValidationError
from core.logging import get_logger
from schemas.common import SuccessResponse
from schemas.discovery import (
    CsvImportRequest,
    CsvImportResponse,
    CsvParseResponse,
)

logger = get_logger(__name__)
router = APIRouter()

_MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB — generous for a leads CSV
                                     # (tens of thousands of rows), tight
                                     # enough to reject an accidental
                                     # wrong-file upload fast rather than
                                     # parsing something enormous
_MAX_IMPORT_ROWS = 15


@router.post(
    "/import/parse",
    response_model=SuccessResponse[CsvParseResponse],
    tags=["Pipeline"],
)
async def parse_csv(
    file: UploadFile = File(...),
    fingerprint: str = Depends(require_fingerprint),
) -> SuccessResponse[CsvParseResponse]:
    """
    Parse an uploaded CSV and report how many rows will be processed.

    Pure parsing — no search, crawl, or Groq calls happen here, so this
    endpoint has no cost and can be called freely while the person is
    still deciding whether to proceed.
    """
    from services.import_service import ImportService

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream", None) and not (
        file.filename or ""
    ).lower().endswith(".csv"):
        raise InputValidationError("File must be a .csv file")

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise InputValidationError(f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")

    rows = ImportService.parse_csv(contents)
    total_rows = len(rows)
    will_process = min(total_rows, _MAX_IMPORT_ROWS)

    logger.info(
        "csv_parsed",
        fingerprint=fingerprint[:8],
        filename=file.filename,
        total_rows=total_rows,
        will_process=will_process,
    )

    return SuccessResponse(
        data=CsvParseResponse(
            rows=rows,
            total_rows=total_rows,
            will_process=will_process,
            capped=total_rows > will_process,
            cap=_MAX_IMPORT_ROWS,
        )
    )


@router.post(
    "/import/resolve",
    response_model=SuccessResponse[CsvImportResponse],
    tags=["Pipeline"],
)
async def resolve_csv_import(
    body: CsvImportRequest,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[CsvImportResponse]:
    """
    Resolve and validate the (already user-confirmed) rows, then persist
    them exactly like Discovery does. The frontend calls this only after
    showing the /parse cap confirmation — but the cap is re-enforced here
    regardless, since a request body is never trusted for something that
    spends real search-provider budget.
    """
    from services.import_service import ImportService

    cache = get_cache(db)
    service = ImportService(db=db, cache=cache)

    result = await service.resolve_and_import(
        rows=body.rows, search_id=body.search_id, icp_id=body.icp_id
    )

    logger.info(
        "csv_import_request_complete",
        search_id=body.search_id,
        rows_submitted=result.rows_submitted,
        rows_processed=result.rows_processed,
        resolved=result.total_found,
        unresolved=len(result.unresolved),
    )

    return SuccessResponse(data=result, meta={"search_id": body.search_id})