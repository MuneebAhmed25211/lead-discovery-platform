"""
schemas/score.py
────────────────
Lead scoring schemas.

The score is 100 points across 6 dimensions.
Every dimension includes the raw score, reasoning text, and
the IDs of evidence records that drove that sub-score.

This makes every score fully explainable to the user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from schemas.common import AppBaseModel, UUIDModel


class ScoreCategory(StrEnum):
    EXCELLENT = "Excellent"   # 90–100
    STRONG    = "Strong"      # 75–89
    MODERATE  = "Moderate"    # 60–74
    WEAK      = "Weak"        # 0–59


def categorize_score(total: int) -> ScoreCategory:
    if total >= 90:
        return ScoreCategory.EXCELLENT
    elif total >= 75:
        return ScoreCategory.STRONG
    elif total >= 60:
        return ScoreCategory.MODERATE
    else:
        return ScoreCategory.WEAK


# ── Score Dimension ───────────────────────────────────────────────────────────

class ScoreDimension(AppBaseModel):
    """A single scoring dimension with explanation."""

    score: int
    max_score: int
    reason: str
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="UUIDs of evidence records that drove this sub-score.",
    )

    @property
    def percentage(self) -> float:
        if self.max_score == 0:
            return 0.0
        return round((self.score / self.max_score) * 100, 1)


# ── Full Score ────────────────────────────────────────────────────────────────

class LeadScore(AppBaseModel):
    """
    Complete 100-point lead score with full breakdown.

    Dimensions and max points:
      icp_match              0–25   keyword + signal overlap with ICP
      industry_fit           0–20   direct industry list match
      company_size           0–15   employee count vs ICP target range    (reduced from 20 — size matters less than fit)
      growth_signals         0–20   blog activity, hiring, expansion signals
      hiring_signals         0–10   active job postings found
      automation_opportunity 0–10   manual-process keywords detected

    Total: 100 points
    """

    company_id: str

    icp_match:              ScoreDimension   # max 25
    industry_fit:           ScoreDimension   # max 20
    company_size:           ScoreDimension   # max 15
    growth_signals:         ScoreDimension   # max 20
    hiring_signals:         ScoreDimension   # max 10
    automation_opportunity: ScoreDimension   # max 10

    total_score: int
    score_category: ScoreCategory
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def max_possible(cls) -> int:
        return 25 + 20 + 15 + 20 + 10 + 10   # = 100

    def to_flat_dict(self) -> dict:
        """Flat representation for CSV export."""
        return {
            "total_score": self.total_score,
            "score_category": self.score_category,
            "icp_match": self.icp_match.score,
            "industry_fit": self.industry_fit.score,
            "company_size": self.company_size.score,
            "growth_signals": self.growth_signals.score,
            "hiring_signals": self.hiring_signals.score,
            "automation_opportunity": self.automation_opportunity.score,
        }


# ── Database Row ──────────────────────────────────────────────────────────────

class LeadScoreRecord(UUIDModel):
    """Supabase `lead_scores` table row."""

    company_id: UUID
    icp_match: int
    industry_fit: int
    company_size: int
    growth_signals: int
    hiring_signals: int
    automation_opportunity: int
    total_score: int
    score_category: str
    score_breakdown: dict    # the full ScoreDimension breakdown as JSON
    evidence_used: list[str]
    scored_at: datetime


# ── API Request/Response ──────────────────────────────────────────────────────

class ScoreRequest(AppBaseModel):
    """POST /api/v1/scoring/score — score one company against one ICP."""
    company_id: str
    icp_id: str


class ScoreBatchRequest(AppBaseModel):
    """POST /api/v1/scoring/batch — score all complete companies for a search."""
    search_id: str
    icp_id: str
    min_research_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Skip companies whose data_confidence is below this threshold.",
    )


class ScoreResponse(AppBaseModel):
    """Response after scoring a company."""
    score: LeadScore
    cached: bool = False


class ScoreBatchResponse(AppBaseModel):
    """Response after batch scoring."""
    scored: int
    skipped: int
    total: int
    scores: list[LeadScore]
