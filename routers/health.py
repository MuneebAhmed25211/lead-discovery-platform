"""
routers/health.py
─────────────────
GET /api/v1/health

Returns status of all downstream dependencies.
Exempt from rate limiting and fingerprint checks.
Used by Render health checks and frontend degraded-service banners.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from core.config import settings
from core.logging import get_logger
from core.supabase import health_check as supabase_health
from schemas.common import HealthResponse, ServiceHealth

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    services: dict[str, ServiceHealth] = {}

    db_result = supabase_health()
    services["supabase"] = ServiceHealth(
        status=db_result["status"], detail=db_result.get("detail", "")
    )

    services["groq"] = await _check_groq()
    services["search"] = _check_search_providers()

    all_ok = all(s.status == "ok" for s in services.values())
    db_ok = services["supabase"].status == "ok"
    overall = "ok" if all_ok else ("error" if not db_ok else "degraded")

    return HealthResponse(status=overall, environment=settings.APP_ENV, services=services)


async def _check_groq() -> ServiceHealth:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            )
            return ServiceHealth(status="ok") if r.status_code == 200 else ServiceHealth(status="error", detail=f"HTTP {r.status_code}")
    except Exception as exc:
        return ServiceHealth(status="error", detail=str(exc)[:100])


def _check_search_providers() -> ServiceHealth:
    configured = [
        n for n, k in [("serper", settings.SERPER_API_KEY), ("tavily", settings.TAVILY_API_KEY), ("brave", settings.BRAVE_API_KEY)]
        if k
    ]
    if configured:
        return ServiceHealth(status="ok", detail=f"providers: {', '.join(configured)}")
    return ServiceHealth(status="error", detail="No search provider keys configured")
