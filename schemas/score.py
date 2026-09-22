"""
schemas/score.py
────────────────
Lead scoring schemas.

The score is 100 points across 6 dimensions.
Every dimension includes the raw score, reasoning text, and
the IDs of evidence records that drove that sub-score.

This makes every score fully explainable to the user.

Dimension weights and score bands below were recalibrated from live
usage data across ~6 independent ICPs (healthcare/manufacturing/
logistics, e-commerce, local-service, automation-for-startups), all of
which clustered around a 40-50/100 ceiling even for companies later
confirmed to be genuinely plausible fits. Root cause: icp_match (was
25 pts) requires a company's own marketing copy to echo the ICP's
free-form pain_points/positive_signals sentences — a bar real company
websites almost never clear regardless of true fit (e.g. a real,
right-sized, actively-hiring law firm scored 0/25 on this dimension
alone, dragging its total to 45/100 despite otherwise-correct industry
and size matches). Weight shifted from icp_match toward the dimensions
that reliably extract from real evidence (industry_fit, company_size,
hiring_signals). Bands lowered to match the realistic achievable range
under the new weights — a first-pass recalibration, expected to be
tuned further as more live data comes in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from schemas.common import AppBaseModel, UUIDModel


class ScoreCategory(StrEnum):
    EXCELLENT = "Excellent"   # 85–100
    STRONG    = "Strong"      # 70–84
    MODERATE  = "Moderate"    # 55–69
    WEAK      = "Weak"        # 0–54


def categorize_score(total: int) -> ScoreCategory:
    if total >= 85:
        return ScoreCategory.EXCELLENT
    elif total >= 70:
        return ScoreCategory.STRONG
    elif total >= 55:
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

    Dimensions and max points (recalibrated — see module docstring):
      icp_match               0–15   keyword + signal overlap with ICP (reduced from 25 —
                                      structurally underscores most real companies)
      industry_fit            0–25   direct industry list match (increased from 20 — reliable)
      company_size            0–20   employee count vs ICP target range (increased from 15)
      growth_signals          0–20   blog activity, hiring, expansion signals (unchanged)
      hiring_signals          0–15   active job postings found (increased from 10 — reliable)
      automation_opportunity  0–5    manual-process keywords detected (reduced from 10 — weak signal)

    Total: 100 points (before disqualifier deduction)

    disqualifiers is a separate, always-visible dimension representing
    points DEDUCTED for ICP disqualifying_signals matched against the
    company's evidence (e.g. "is itself a software/AI company"). Its
    score is always <= 0 and its max_score is 0 — it is not part of the
    100-point ceiling — but it is exposed here with its own reason text
    so a deduction is never a silent, unexplained subtraction from the
    total. Every point awarded or deducted must be traceable to a rule.
    """

    company_id: str

    icp_match:              ScoreDimension   # max 15
    industry_fit:           ScoreDimension   # max 25
    company_size:           ScoreDimension   # max 20
    growth_signals:         ScoreDimension   # max 20
    hiring_signals:         ScoreDimension   # max 15
    automation_opportunity: ScoreDimension   # max 5
    disqualifiers:          ScoreDimension   # max 0, score <= 0 — deduction only

    total_score: int
    score_category: ScoreCategory
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def max_possible(cls) -> int:
        return 15 + 25 + 20 + 20 + 15 + 5   # = 100 (disqualifiers contributes 0 to the max)

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
            "disqualifiers": self.disqualifiers.score,
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