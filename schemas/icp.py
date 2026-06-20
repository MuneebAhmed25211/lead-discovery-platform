"""
schemas/icp.py
──────────────
Pydantic schemas for ICP (Ideal Customer Profile) generation.

The ICP is the first output of the pipeline and drives everything else:
  - search queries used for discovery
  - scoring weights
  - qualification prompt context
  - outreach angle selection
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator

from schemas.common import AppBaseModel, TimestampedModel, UUIDModel


# ── Request ───────────────────────────────────────────────────────────────────

class ICPGenerateRequest(AppBaseModel):
    """
    What the user provides to kick off ICP generation.

    service_description: Free-text description of what the user sells.
    Examples:
      - "I build AI automation systems for e-commerce companies"
      - "I'm a marketing agency specialising in SaaS lead generation"
      - "I provide bookkeeping and tax services for small businesses"
    """

    service_description: Annotated[
        str,
        Field(
            min_length=20,
            max_length=500,
            description="Describe what you sell and who you typically work with.",
        ),
    ]
    search_id: str = Field(
        ...,
        description="UUID of the search session this ICP belongs to.",
    )

    @field_validator("service_description")
    @classmethod
    def strip_and_clean(cls, v: str) -> str:
        cleaned = " ".join(v.split())  # normalise whitespace
        return cleaned


# ── Core ICP Object ───────────────────────────────────────────────────────────

class ICP(AppBaseModel):
    """
    Structured Ideal Customer Profile output from Groq.

    All list fields are non-empty — Groq is instructed to always provide
    at least one value. The scoring engine reads these directly.
    """

    # Who we're targeting
    target_industries: list[str] = Field(
        ...,
        min_length=1,
        description="e.g. ['e-commerce', 'SaaS', 'retail']",
    )
    target_company_sizes: list[str] = Field(
        ...,
        min_length=1,
        description="e.g. ['11-50', '51-200']",
    )
    target_locations: list[str] = Field(
        ...,
        min_length=1,
        description="e.g. ['United States', 'United Kingdom', 'Germany']",
    )

    # What problems they have
    pain_points: list[str] = Field(
        ...,
        min_length=1,
        description="Specific operational pain points this service solves.",
    )
    buying_triggers: list[str] = Field(
        ...,
        min_length=1,
        description="Events that make this company likely to buy right now.",
    )

    # What to look for when researching
    positive_signals: list[str] = Field(
        ...,
        min_length=1,
        description="Signals that indicate a strong fit.",
    )
    disqualifying_signals: list[str] = Field(
        default_factory=list,
        description="Signals that should lower the score or disqualify.",
    )

    # Discovery
    search_queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Search queries to feed the discovery engine. Max 5.",
    )

    # Human-readable summary for display
    summary: str = Field(
        ...,
        description="One-paragraph summary of the ideal customer.",
    )


# ── Database Row ──────────────────────────────────────────────────────────────

class ICPRecord(UUIDModel, TimestampedModel):
    """ICP as stored in the Supabase `icps` table."""

    search_id: str
    description_hash: str
    target_industries: list[str]
    target_company_sizes: list[str]
    target_locations: list[str]
    pain_points: list[str]
    buying_triggers: list[str]
    positive_signals: list[str]
    disqualifying_signals: list[str]
    search_queries: list[str]
    summary: str
    raw_json: dict  # the full Groq response, stored for debugging


# ── Response ──────────────────────────────────────────────────────────────────

class ICPGenerateResponse(AppBaseModel):
    """API response for POST /api/v1/icp/generate."""

    icp_id: str
    search_id: str
    icp: ICP
    cached: bool = Field(
        default=False,
        description="True if this ICP was served from cache (no Groq call made).",
    )
