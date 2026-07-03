"""
schemas/company.py
──────────────────
Company schemas across three lifecycle stages:

  1. RawCompany   — returned by discovery (search snippets, P3 data only)
  2. Company      — after research (enriched with P1+P2 evidence)
  3. CompanyRecord — the database row shape

The distinction is intentional: a RawCompany cannot be scored.
Only a researched Company with a populated EvidenceBundle can be scored.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, HttpUrl, field_validator

from schemas.common import AppBaseModel, TimestampedModel, UUIDModel


# ── Research Status ───────────────────────────────────────────────────────────

class ResearchStatus(StrEnum):
    PENDING      = "pending"
    IN_PROGRESS  = "in_progress"
    COMPLETE     = "complete"
    FAILED       = "failed"
    SKIPPED      = "skipped"   # e.g. robots.txt blocked all pages


# ── Stage 1: Discovery Output ─────────────────────────────────────────────────

class RawCompany(AppBaseModel):
    """
    A company as returned by the search provider.

    Contains only P3-trust data: search title, URL, and snippet.
    This is a discovery hint — it drives the research crawl
    but is NEVER used directly in scoring or qualification.
    """

    name: str = Field(..., description="Company name extracted from search result title")
    domain: str = Field(..., description="Cleaned root domain, e.g. 'acme.com'")
    website: str = Field(..., description="Full URL of the company homepage")
    source_url: str = Field(..., description="The search result URL (may differ from homepage)")
    snippet: str = Field(default="", description="Search result snippet — P3 data")
    provider: str = Field(..., description="Which search provider returned this result")

    @field_validator("domain")
    @classmethod
    def clean_domain(cls, v: str) -> str:
        """Strip scheme, www, and trailing slash."""
        v = v.lower().strip()
        for prefix in ("https://", "http://", "www."):
            if v.startswith(prefix):
                v = v[len(prefix):]
        return v.rstrip("/")


# ── Stage 2: Researched Company ───────────────────────────────────────────────

class Company(AppBaseModel):
    """
    A company after research has completed.

    All fields below are populated from P1 (company site) and P2
    (business directories) evidence. None of these values come from
    search snippets alone.
    """

    id: str
    search_id: str
    name: str
    domain: str
    website: str

    # Discovery origin (P3 — kept for audit trail only)
    discovery_source_url: str = ""
    discovery_snippet: str = ""

    # Resolved from P1+P2 evidence
    location: str = ""
    industry: str = ""
    employee_range: str = ""                # e.g. "11-50", "51-200", "201-500"
    services: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    hiring_signals: bool = False
    job_titles_found: list[str] = Field(default_factory=list)
    social_profiles: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'linkedin': 'https://...', 'twitter': 'https://...'}"
    )
    contact_pages: list[str] = Field(default_factory=list)
    contact_emails: list[str] = Field(default_factory=list)
    contact_phones: list[str] = Field(default_factory=list)
    booking_detected: bool = False

    # Trust metadata
    p1_pages_crawled: list[str] = Field(default_factory=list)
    p1_pages_failed: list[str] = Field(default_factory=list)
    data_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    research_status: ResearchStatus = ResearchStatus.PENDING
    researched_at: datetime | None = None

    @property
    def is_researchable(self) -> bool:
        """True if we have enough data to attempt a crawl."""
        return bool(self.domain and self.website)

    @property
    def employee_count_midpoint(self) -> int:
        """
        Parse employee_range to a numeric midpoint for scoring.
        Returns 0 if unparseable.
        """
        if not self.employee_range:
            return 0
        parts = self.employee_range.replace("+", "").split("-")
        try:
            nums = [int(p.strip()) for p in parts if p.strip().isdigit()]
            return sum(nums) // len(nums) if nums else 0
        except (ValueError, ZeroDivisionError):
            return 0


# ── Database Row ──────────────────────────────────────────────────────────────

class CompanyRecord(UUIDModel, TimestampedModel):
    """Supabase `companies` table row shape."""

    search_id: str  # searches.id is TEXT in the database, not UUID
    name: str
    domain: str
    website: str
    discovery_source_url: str
    discovery_snippet: str
    location: str
    industry: str
    employee_range: str
    services: list[str]
    technologies: list[str]
    hiring_signals: bool
    job_titles_found: list[str]
    social_profiles: dict
    contact_pages: list[str]
    contact_emails: list[str]
    contact_phones: list[str]
    booking_detected: bool
    p1_pages_crawled: list[str]
    p1_pages_failed: list[str]
    data_confidence: float
    research_status: str
    researched_at: datetime | None


# ── API Request/Response ──────────────────────────────────────────────────────

class ResearchRequest(AppBaseModel):
    """POST /api/v1/research/company — research a single discovered company."""

    discovered_company_id: str = Field(..., description="UUID from discovered_companies table")


class ResearchBatchRequest(AppBaseModel):
    """POST /api/v1/research/batch — research all unpromoted candidates for a search."""

    search_id: str
    max_companies: int = Field(default=20, ge=1, le=50)


class ResearchResponse(AppBaseModel):
    """Response after researching a single company."""

    company: Company
    evidence_count: int
    p1_pages_crawled: list[str]
    p1_pages_failed: list[str]
    data_confidence: float
    cached: bool = False
