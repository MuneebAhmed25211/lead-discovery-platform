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

ProviderName = Literal["serper", "tavily", "csv_import"]


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


# ── CSV import (Path 2) ──────────────────────────────────────────────────────
#
# Discovery's normal path generates candidate domains from ICP search
# queries. This path instead takes a user-supplied list of company names
# (optionally with a website already attached) and resolves/validates each
# one the same way — reusing validate_candidate and the SAME
# DiscoveredCompany shape, so every downstream stage (Research, Score,
# Qualify, and every result-page component) is completely unchanged. The
# only new work is: parsing a CSV, and — for rows with no website supplied
# — a single targeted search per company to find one, through the same
# Serper→Tavily router Discovery already uses.
#
# Two-step flow, matching the UI decision to show the cap BEFORE spending
# any search/crawl/Groq budget:
#   1. POST /import/parse   — parse the file only, no cost. Returns every
#      row plus how many will actually be processed, so the frontend can
#      show the "first N of your M leads" confirmation before proceeding.
#   2. POST /import/resolve — the user confirmed; resolve + validate the
#      (already-capped) rows and persist them exactly like Discovery does.

_MAX_IMPORT_ROWS = 15  # see services/import_service.py for the full
                       # rationale (this is a deliberate demo cap, not a
                       # technical limit — surfaced to the user, not hidden)


class ImportedLeadRow(AppBaseModel):
    """One parsed row from a user-uploaded CSV, before resolution."""

    raw_name: str = Field(..., description="Company name as given in the CSV")
    raw_website: str = Field(
        default="", description="Website/domain from the CSV, if the column was present and populated"
    )

    @field_validator("raw_name")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class CsvParseResponse(AppBaseModel):
    """POST /api/v1/import/parse response data."""

    rows: list[ImportedLeadRow]
    total_rows: int
    will_process: int = Field(
        description="How many rows will actually be resolved/researched if the user proceeds (min(total_rows, cap))"
    )
    capped: bool = Field(description="True if total_rows > will_process")
    cap: int = _MAX_IMPORT_ROWS


class CsvImportRequest(AppBaseModel):
    """
    POST /api/v1/import/resolve request body.

    `rows` is expected to already be the user-confirmed, capped subset (the
    frontend re-sends exactly what /import/parse returned, sliced to the
    cap) — but the cap is re-enforced server-side regardless of what's
    sent, since the client is never trusted for something that spends
    real search-provider and (downstream) Groq budget.
    """

    search_id: str
    icp_id: str
    rows: list[ImportedLeadRow]


class CsvImportResponse(AppBaseModel):
    """POST /api/v1/import/resolve response data."""

    search_id: str
    companies: list[DiscoveredCompany]
    total_found: int
    rows_submitted: int
    rows_processed: int = Field(description="min(rows_submitted, cap) — the cap enforced server-side")
    unresolved: list[str] = Field(
        default_factory=list,
        description="Company names that couldn't be resolved to a validated website — shown transparently, same treatment as Discovery's/Research's crawler-blocked list, never silently dropped",
    )