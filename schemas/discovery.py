"""
schemas/discovery.py
────────────────────
Pydantic schemas for the Discovery Engine (Phase 3).
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator

from schemas.common import AppBaseModel

ProviderName = Literal["serper", "tavily"]


# ── Provider-level ────────────────────────────────────────────────────────────

class SearchResult(AppBaseModel):
    """
    A single raw result from a search provider, before normalization.

    Both Serper and Tavily clients return a list[SearchResult] from their
    search() method — this is the common interface the router and service
    layer depend on, regardless of which provider produced it.
    """

    title: str = Field(..., description="Result title, typically the page <title> or company name guess")
    url: str = Field(..., description="The result URL — may be homepage or a subpage")
    snippet: str = Field(default="", description="Search result snippet text — P3 trust data")
    provider: ProviderName

    @field_validator("title", "url")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class ProviderSearchOutcome(AppBaseModel):
    """
    The outcome of attempting a search through the provider router for ONE query.

    Captures not just results but what happened during the attempt — used by
    the discovery service to decide whether to log a fallback event and to
    build the final partial_failure flag in the API response.
    """

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    provider_used: ProviderName | Literal["none"] = "none"
    fallback_triggered: bool = False
    cached: bool = False
    succeeded: bool = True
    error: str = ""


# ── Service-level ──────────────────────────────────────────────────────────────

class DiscoveredCompany(AppBaseModel):
    """
    A normalized, deduplicated company candidate.

    `id` is always a non-empty string after persistence. Before DB insert it
    is pre-generated as a UUID so even fallback return paths carry a real ID.
    `domain` is the identity key — see utils/dedup.py for normalization rules.
    """

    id: str = Field(default_factory=lambda: str(_uuid.uuid4()))
    search_id: str
    icp_id: str
    name: str
    domain: str
    website: str
    source_url: str
    snippet: str = ""
    provider: ProviderName
    matched_query: str
    linkedin_url: str = ""   # enrichment — linkedin.com/company/{slug} only
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── API-level ──────────────────────────────────────────────────────────────────

class DiscoveryRequest(AppBaseModel):
    """POST /api/v1/discovery/search request body."""

    search_id: str = Field(..., description="Identifier of the search session (searches.id, stored as text)")
    icp_id: str = Field(..., description="UUID of the ICP to discover companies for")


class DiscoveryResponse(AppBaseModel):
    """POST /api/v1/discovery/search response data."""

    search_id: str
    companies: list[DiscoveredCompany]
    total_found: int
    queries_executed: int
    queries_failed: int
    partial_failure: bool = Field(
        default=False,
        description=(
            "True only when provider failures (not legitimate zero-result "
            "queries) caused a reduction in results. See discovery_service "
            "for the exact distinction."
        ),
    )
