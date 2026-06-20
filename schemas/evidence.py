"""
schemas/evidence.py
───────────────────
Schemas for the evidence system — the core of the trust hierarchy.

Every fact about a company is an Evidence record with:
  - where it came from (source_url)
  - how trustworthy that source is (trust_level 1/2/3)
  - what it says (field_name + value)
  - how confident we are (confidence 0.0–1.0)

The scoring engine only uses evidence with trust_level <= 2.
The qualification Groq prompt only receives trust_level == 1 evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from typing import Annotated
from uuid import UUID

from pydantic import Field

from schemas.common import AppBaseModel, UUIDModel


# ── Trust Levels ──────────────────────────────────────────────────────────────

class TrustLevel(IntEnum):
    """
    Source trust hierarchy from Phase 1 architecture.

    P1 = direct evidence from the company's own website.
         Drives scoring and qualification.
    P2 = business databases / directories.
         Used for enrichment only.
    P3 = search snippets, news, third-party articles.
         Discovery hints only — never used in scoring.
    """

    P1 = 1  # Company's own website
    P2 = 2  # Business databases / directories
    P3 = 3  # External hints — discovery only


class SourceType(str):
    """
    String constants for source_type field.
    Using str subclass (not StrEnum) for Pydantic v2 compatibility.
    """
    COMPANY_HOMEPAGE    = "company_homepage"
    COMPANY_ABOUT       = "company_about"
    COMPANY_SERVICES    = "company_services"
    COMPANY_CAREERS     = "company_careers"
    COMPANY_BLOG        = "company_blog"
    COMPANY_PRESS       = "company_press"
    COMPANY_LEADERSHIP  = "company_leadership"
    CRUNCHBASE          = "crunchbase"
    ANGELLIST           = "angellist"
    BUSINESS_DIRECTORY  = "business_directory"
    SEARCH_SNIPPET      = "search_snippet"
    NEWS_MENTION        = "news_mention"
    THIRD_PARTY_ARTICLE = "third_party_article"

    # Mapping from source_type → trust level
    TRUST_MAP: dict[str, int] = {
        "company_homepage":    1,
        "company_about":       1,
        "company_services":    1,
        "company_careers":     1,
        "company_blog":        1,
        "company_press":       1,
        "company_leadership":  1,
        "crunchbase":          2,
        "angellist":           2,
        "business_directory":  2,
        "search_snippet":      3,
        "news_mention":        3,
        "third_party_article": 3,
    }

    @classmethod
    def trust_level_for(cls, source_type: str) -> int:
        return cls.TRUST_MAP.get(source_type, 3)


# ── Evidence Schema ───────────────────────────────────────────────────────────

class Evidence(AppBaseModel):
    """
    A single verifiable fact about a company, with full provenance.

    field_name examples:
      'industry', 'employee_count', 'location', 'service',
      'technology', 'hiring_signal', 'contact_email', 'social_linkedin'
    """

    company_id: str
    source_url: str
    source_type: str        # one of SourceType constants
    trust_level: int        # 1, 2, or 3
    field_name: str
    value: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class EvidenceRecord(UUIDModel):
    """Evidence as stored in the `evidence` table."""

    company_id: UUID
    source_url: str
    source_type: str
    trust_level: int
    field_name: str
    value: str
    confidence: float
    extracted_at: datetime


class EvidenceBundle(AppBaseModel):
    """
    All evidence collected for a single company, grouped by trust level.
    This is what gets passed to the scoring engine and qualification prompt.
    """

    company_id: str
    p1_evidence: list[Evidence] = Field(default_factory=list)
    p2_evidence: list[Evidence] = Field(default_factory=list)
    # p3 is NOT included — it's used for discovery only

    @property
    def all_evidence(self) -> list[Evidence]:
        return self.p1_evidence + self.p2_evidence

    @property
    def overall_confidence(self) -> float:
        """
        Weighted average confidence across all evidence.
        P1 carries 1.0 weight, P2 carries 0.6 weight.
        Returns 0.0 if no evidence.
        """
        if not self.all_evidence:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for e in self.p1_evidence:
            weight = 1.0
            weighted_sum += e.confidence * weight
            total_weight += weight

        for e in self.p2_evidence:
            weight = 0.6
            weighted_sum += e.confidence * weight
            total_weight += weight

        return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0

    def get_values(self, field_name: str, max_trust: int = 2) -> list[str]:
        """
        Retrieve all values for a given field_name up to the given trust level.
        Used by the scoring engine: score_company(..., bundle.get_values('industry'))
        """
        return [
            e.value
            for e in self.all_evidence
            if e.field_name == field_name and e.trust_level <= max_trust
        ]

    def p1_facts_for_prompt(self) -> dict[str, list[str]]:
        """
        Return a dict of field_name → [values] containing ONLY P1 evidence.
        This is what gets serialised into the qualification Groq prompt.
        """
        facts: dict[str, list[str]] = {}
        for e in self.p1_evidence:
            facts.setdefault(e.field_name, [])
            if e.value not in facts[e.field_name]:
                facts[e.field_name].append(e.value)
        return facts
