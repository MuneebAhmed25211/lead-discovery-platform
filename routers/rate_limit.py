"""
routers/rate_limit.py
─────────────────────
GET /api/v1/rate-limit/status

Returns the current rate limit state for a fingerprint.
Called by the frontend on page load to decide whether to show the upgrade banner.
Exempt from the rate-limit middleware (it's just a status check).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from supabase import Client

from core.dependencies import get_client_ip, get_db, require_fingerprint
from middleware.rate_limit import RateLimitService
from schemas.common import SuccessResponse
from schemas.rate_limit import RateLimitStatusResponse

router = APIRouter()


@router.get("/rate-limit/status", tags=["System"])
async def get_rate_limit_status(
    request: Request,
    fingerprint: str = Depends(require_fingerprint),
    db: Client = Depends(get_db),
) -> SuccessResponse[RateLimitStatusResponse]:
    ip = get_client_ip(request)
    service = RateLimitService(db)
    status = await service.get_status(fingerprint=fingerprint, ip=ip)
    return SuccessResponse(data=status)
