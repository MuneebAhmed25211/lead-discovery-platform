"""
core/config.py
──────────────
Single source of truth for all application configuration.

Uses pydantic-settings v2 which reads values from:
  1. Environment variables (highest priority)
  2. A .env file in the project root
  3. Field default values (lowest priority)

All config is validated at startup — the application refuses to start
if required values are missing or invalid. This prevents runtime surprises.

Usage:
    from core.config import settings
    print(settings.SUPABASE_URL)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_SECRET_KEY: str = Field(
        ...,
        min_length=32,
        description="Random secret used for internal signing. Min 32 chars.",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS into a list."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # ── Supabase ─────────────────────────────────────────────────────────────
    SUPABASE_URL: AnyHttpUrl = Field(..., description="Your Supabase project URL")
    SUPABASE_ANON_KEY: str = Field(..., min_length=10)
    SUPABASE_SERVICE_KEY: str = Field(
        ...,
        min_length=10,
        description="Service role key — backend only, never expose to browser",
    )

    # ── Groq ─────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., min_length=10)
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = Field(default=2048, ge=256, le=8192)
    GROQ_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── Search Providers ─────────────────────────────────────────────────────
    SERPER_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    BRAVE_API_KEY: str = ""
    SEARCH_PROVIDER_ORDER: str = "serper,tavily,brave"

    @property
    def search_provider_order(self) -> list[str]:
        return [p.strip() for p in self.SEARCH_PROVIDER_ORDER.split(",") if p.strip()]

    @model_validator(mode="after")
    def at_least_one_search_provider(self) -> "Settings":
        keys = {
            "serper": self.SERPER_API_KEY,
            "tavily": self.TAVILY_API_KEY,
            "brave": self.BRAVE_API_KEY,
        }
        configured = [name for name, key in keys.items() if key]
        if not configured:
            raise ValueError(
                "At least one search provider API key must be set: "
                "SERPER_API_KEY, TAVILY_API_KEY, or BRAVE_API_KEY"
            )
        return self

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_MAX_SEARCHES: int = Field(default=3, ge=1, le=100)
    RATE_LIMIT_WINDOW_DAYS: int = Field(default=30, ge=1, le=365)
    RATE_LIMIT_CONTACT_URL: str = "https://www.linkedin.com/in/muneebahmedai"

    # ── Cache TTLs (in hours) ─────────────────────────────────────────────────
    CACHE_ICP_TTL_HOURS: int = Field(default=24, ge=1)
    CACHE_DISCOVERY_TTL_HOURS: int = Field(default=2, ge=1)
    CACHE_P1_CRAWL_TTL_HOURS: int = Field(default=72, ge=1)
    CACHE_P2_ENRICH_TTL_HOURS: int = Field(default=24, ge=1)
    CACHE_SCORE_TTL_HOURS: int = Field(default=48, ge=1)
    CACHE_QUALIFY_TTL_HOURS: int = Field(default=48, ge=1)
    CACHE_OUTREACH_TTL_HOURS: int = Field(default=48, ge=1)

    # ── Crawler ───────────────────────────────────────────────────────────────
    CRAWLER_TIMEOUT_SECONDS: int = Field(default=10, ge=3, le=30)
    CRAWLER_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    CRAWLER_RESPECT_ROBOTS: bool = True
    CRAWLER_USER_AGENT: str = (
        "LeadDiscoveryBot/1.0 (portfolio demo; contact muneebahmedai@gmail.com)"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    lru_cache ensures we only parse env vars once per process lifetime,
    which is important for startup performance and test isolation.
    """
    return Settings()


# Module-level singleton — import this everywhere
settings: Settings = get_settings()
