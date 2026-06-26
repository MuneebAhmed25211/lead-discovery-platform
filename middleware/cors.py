"""
middleware/cors.py
──────────────────
CORS configuration for the FastAPI application.

In production: only the Vercel frontend origin is allowed.
In development: localhost:3000 is always added.

Never uses wildcard (*) — all API keys are server-side but we still
want to prevent direct browser access from unauthorized origins.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings


def setup_cors(app: FastAPI) -> None:
    origins = settings.cors_origins

    # Always allow localhost in development
    if settings.is_development and "http://localhost:3000" not in origins:
        origins = ["http://localhost:3000"] + origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Fingerprint", "Authorization"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
        max_age=86400,  # Cache preflight for 24 hours
    )
