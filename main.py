"""
main.py
───────
FastAPI application entry point.

Responsibilities:
  1. Application factory (create_app)
  2. Lifespan handler — startup and shutdown hooks
  3. Middleware registration (order matters — CORS first, then logging, then rate limit)
  4. Router registration under /api/v1
  5. Global exception handler — converts LeadDiscoveryError → clean JSON responses
  6. Pydantic validation error handler — converts 422s to our error envelope

To run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Production (Render start command):
    uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.config import settings
from core.errors import LeadDiscoveryError
from core.logging import get_logger, setup_logging
from core.supabase import init_supabase
from middleware.cors import setup_cors
from middleware.logging_middleware import RequestLoggingMiddleware
from middleware.rate_limit import RateLimitMiddleware

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup and shutdown lifecycle manager.

    Startup:
      - Configure structured logging
      - Initialise Supabase clients
      - Log application boot info

    Shutdown:
      - Graceful close (currently nothing to tear down explicitly)
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    setup_logging()
    logger.info(
        "app_starting",
        environment=settings.APP_ENV,
        version="1.0.0",
        groq_model=settings.GROQ_MODEL,
    )

    init_supabase()
    logger.info("app_ready", host=settings.APP_HOST, port=settings.APP_PORT)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("app_shutting_down")


# ── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Lead Discovery Platform API",
        description=(
            "AI-powered lead discovery and qualification platform. "
            "Built by Muneeb Ahmed — Full Stack AI Developer."
        ),
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware (registered in reverse order — last registered runs first) ─
    setup_cors(app)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    _register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    from routers import (
        discovery,
        export,
        health,
        icp,
        outreach,
        qualification,
        rate_limit,
        research,
        scoring,
    )

    prefix = "/api/v1"

    # System routes (no fingerprint required)
    app.include_router(health.router, prefix=prefix)
    app.include_router(rate_limit.router, prefix=prefix)

    # Pipeline routes (fingerprint required — enforced per-route via Depends)
    app.include_router(icp.router, prefix=prefix)
    app.include_router(discovery.router, prefix=prefix)
    app.include_router(research.router, prefix=prefix)
    app.include_router(scoring.router, prefix=prefix)
    app.include_router(qualification.router, prefix=prefix)
    app.include_router(outreach.router, prefix=prefix)
    app.include_router(export.router, prefix=prefix)


def _register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(LeadDiscoveryError)
    async def lead_discovery_error_handler(
        request: Request, exc: LeadDiscoveryError
    ) -> JSONResponse:
        """
        Convert any LeadDiscoveryError subclass into our standard error envelope.
        The HTTP status and error code come from the exception class itself.
        """
        logger.warning(
            "api_error",
            code=exc.code,
            message=exc.message,
            path=request.url.path,
            status=exc.http_status,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={"success": False, **exc.to_dict()},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Convert Pydantic v2 request validation errors into our error envelope.
        Returns a human-readable summary of which fields failed.
        """
        errors = exc.errors()
        field_errors = {
            " → ".join(str(loc) for loc in err["loc"]): err["msg"]
            for err in errors
        }
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "fields": field_errors,
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all for unhandled exceptions.
        Never expose internal details in production.
        """
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        message = str(exc) if settings.is_development else "An unexpected error occurred."
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_ERROR",
                "message": message,
            },
        )


# ── Module-level app instance ─────────────────────────────────────────────────
app = create_app()


# ── Local development entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower(),
    )
