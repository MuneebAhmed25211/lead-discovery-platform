"""
core/errors.py
──────────────
Custom exception hierarchy for the Lead Discovery Platform.

Design principles:
  - Every exception carries a machine-readable `code` string for the API.
  - HTTP status codes are set here, not scattered across routers.
  - The global exception handler in main.py maps these to JSON responses.
  - Internal errors (database, crawl) never expose raw details to clients.

Exception tree:
    LeadDiscoveryError (base)
    ├── RateLimitExceededError   → 429
    ├── FingerprintMissingError  → 400
    ├── ValidationError          → 422
    ├── NotFoundError            → 404
    ├── GroqError                → 502
    │   └── GroqRateLimitError   → 429
    ├── SearchProviderError      → 502
    │   └── AllProvidersFailedError
    ├── CrawlError               → 502
    │   ├── RobotsDisallowedError → 403
    │   └── CrawlTimeoutError
    ├── CacheError               → 500
    └── DatabaseError            → 500
"""

from __future__ import annotations

from typing import Any


class LeadDiscoveryError(Exception):
    """Base exception for all platform errors."""

    http_status: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            **self.detail,
        }


# ── Rate Limiting ─────────────────────────────────────────────────────────────

class RateLimitExceededError(LeadDiscoveryError):
    """Raised when a fingerprint has exhausted its free search quota."""

    http_status = 429
    code = "RATE_LIMIT_EXCEEDED"

    def __init__(self, used: int, max_allowed: int, contact_url: str) -> None:
        super().__init__(
            f"You have used {used} of {max_allowed} free searches.",
            detail={
                "searches_used": used,
                "searches_allowed": max_allowed,
                "upgrade_url": contact_url,
                "upgrade_message": (
                    "This platform is a live demo of my AI automation work. "
                    "Contact me to build a custom lead generation system for your business."
                ),
            },
        )


class FingerprintMissingError(LeadDiscoveryError):
    """Raised when X-Fingerprint header is absent or malformed."""

    http_status = 400
    code = "FINGERPRINT_MISSING"

    def __init__(self) -> None:
        super().__init__(
            "Request is missing the X-Fingerprint header.",
            detail={"hint": "Include a FingerprintJS visitor ID in X-Fingerprint."},
        )


# ── Input Validation ──────────────────────────────────────────────────────────

class InputValidationError(LeadDiscoveryError):
    """Domain-level validation failure (distinct from Pydantic's built-in)."""

    http_status = 422
    code = "VALIDATION_ERROR"


class NotFoundError(LeadDiscoveryError):
    """Resource not found."""

    http_status = 404
    code = "NOT_FOUND"

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            f"{resource} '{identifier}' not found.",
            detail={"resource": resource, "identifier": identifier},
        )


# ── Groq / AI ─────────────────────────────────────────────────────────────────

class GroqError(LeadDiscoveryError):
    """Groq API call failed after all retries."""

    http_status = 502
    code = "GROQ_ERROR"

    def __init__(self, operation: str, cause: str) -> None:
        super().__init__(
            f"AI service failed during '{operation}'. Please try again.",
            detail={"operation": operation, "cause": cause},
        )


class GroqRateLimitError(GroqError):
    """Groq API rate limit hit (free tier)."""

    http_status = 429
    code = "GROQ_RATE_LIMIT"

    def __init__(self, operation: str) -> None:
        LeadDiscoveryError.__init__(
            self,
            f"AI service is temporarily rate-limited. Retry in a moment.",
            detail={"operation": operation},
        )


# ── Search Providers ──────────────────────────────────────────────────────────

class SearchProviderError(LeadDiscoveryError):
    """A single search provider failed."""

    http_status = 502
    code = "SEARCH_PROVIDER_ERROR"

    def __init__(self, provider: str, cause: str) -> None:
        super().__init__(
            f"Search provider '{provider}' failed.",
            detail={"provider": provider, "cause": cause},
        )


class AllProvidersFailedError(LeadDiscoveryError):
    """Every configured search provider failed for this query."""

    http_status = 502
    code = "ALL_PROVIDERS_FAILED"

    def __init__(self, query: str, errors: dict[str, str]) -> None:
        super().__init__(
            "All search providers are currently unavailable. Please try again.",
            detail={"query": query, "provider_errors": errors},
        )


# ── Crawler ───────────────────────────────────────────────────────────────────

class CrawlError(LeadDiscoveryError):
    """Website crawl failed."""

    http_status = 502
    code = "CRAWL_ERROR"

    def __init__(self, domain: str, url: str, cause: str) -> None:
        super().__init__(
            f"Could not crawl '{url}'.",
            detail={"domain": domain, "url": url, "cause": cause},
        )


class RobotsDisallowedError(CrawlError):
    """robots.txt disallows crawling this URL."""

    http_status = 403
    code = "ROBOTS_DISALLOWED"

    def __init__(self, domain: str, url: str) -> None:
        LeadDiscoveryError.__init__(
            self,
            f"robots.txt disallows crawling '{url}'.",
            detail={"domain": domain, "url": url},
        )


class CrawlTimeoutError(CrawlError):
    """HTTP request timed out during crawl."""

    code = "CRAWL_TIMEOUT"

    def __init__(self, domain: str, url: str, timeout: int) -> None:
        LeadDiscoveryError.__init__(
            self,
            f"Crawl timed out after {timeout}s.",
            detail={"domain": domain, "url": url, "timeout_seconds": timeout},
        )


# ── Infrastructure ────────────────────────────────────────────────────────────

class CacheError(LeadDiscoveryError):
    """Cache read or write failed. Non-fatal — callers should proceed without cache."""

    http_status = 500
    code = "CACHE_ERROR"

    def __init__(self, operation: str, key: str, cause: str) -> None:
        super().__init__(
            f"Cache {operation} failed for key '{key}'.",
            detail={"operation": operation, "key": key, "cause": cause},
        )


class DatabaseError(LeadDiscoveryError):
    """Supabase / Postgres operation failed."""

    http_status = 500
    code = "DATABASE_ERROR"

    def __init__(self, operation: str, table: str, cause: str) -> None:
        super().__init__(
            "A database error occurred. Please try again.",
            detail={"operation": operation, "table": table, "cause": cause},
        )
