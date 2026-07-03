"""
services/scoring_service.py
────────────────────────────
Deterministic lead scoring engine. Zero Groq calls.

Scores a researched company against a generated ICP on 6 dimensions:
  icp_match              0-25
  industry_fit           0-20
  company_size           0-15
  growth_signals         0-20
  hiring_signals         0-10
  automation_opportunity 0-10
  Total:                 100

Every point traces to a specific evidence record and a specific rule.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from supabase import Client

from core.logging import get_logger
from schemas.score import (
    LeadScore,
    ScoreBatchResponse,
    ScoreDimension,
    categorize_score,
)
from utils.text import keyword_overlap

logger = get_logger(__name__)

_MIN_CONFIDENCE_TO_SCORE = 0.10

_AUTOMATION_KEYWORDS = [
    "manual", "spreadsheet", "paper", "phone call", "walk-in",
    "fax", "voicemail", "paper form", "handwritten", "pen and paper",
    "call us", "call to book", "call to schedule", "no online booking",
    "in-person only", "appointment by phone", "leave a message",
    "patient intake form", "clipboard", "front desk",
]

_GROWTH_TECH_SIGNALS = {
    "HubSpot", "Salesforce", "Calendly", "Acuity", "Mindbody",
    "Jane App", "Healthie", "Kareo", "Athenahealth", "ActiveCampaign",
}

_SIZE_TIERS = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]


class ScoringService:

    def __init__(self, db: Client) -> None:
        self._db = db

    def score_company(self, company_id: str, icp_id: str) -> LeadScore:
        company = self._load_company(company_id)
        icp = self._load_icp(icp_id)
        evidence = self._load_evidence(company_id)
        score = self._compute_score(company, icp, evidence)
        self._persist_score(score)
        logger.info("company_scored", domain=company.get("domain"),
                    total=score.total_score, category=score.score_category)
        return score

    def score_batch(self, search_id: str, icp_id: str,
                    min_research_confidence: float = 0.0) -> ScoreBatchResponse:
        companies = self._load_complete_companies(search_id)
        icp = self._load_icp(icp_id)
        scores: list[LeadScore] = []
        skipped = 0

        for company in companies:
            company_id = company["id"]
            if self._already_scored(company_id):
                skipped += 1
                continue
            if company.get("data_confidence", 0.0) < max(_MIN_CONFIDENCE_TO_SCORE, min_research_confidence):
                skipped += 1
                continue
            try:
                evidence = self._load_evidence(company_id)
                score = self._compute_score(company, icp, evidence)
                self._persist_score(score)
                scores.append(score)
                logger.info("batch_scored", domain=company.get("domain"), total=score.total_score)
            except Exception as exc:
                logger.warning("batch_score_failed", company_id=company_id, error=str(exc))
                skipped += 1

        return ScoreBatchResponse(
            scored=len(scores), skipped=skipped, total=len(companies),
            scores=sorted(scores, key=lambda s: s.total_score, reverse=True),
        )

    def _compute_score(self, company: dict, icp: dict,
                        evidence: list[dict]) -> LeadScore:
        ev_by_field: dict[str, list[dict]] = {}
        for ev in evidence:
            ev_by_field.setdefault(ev["field_name"], []).append(ev)

        icp_dim    = self._score_icp_match(company, icp, ev_by_field)
        ind_dim    = self._score_industry_fit(company, icp, ev_by_field)
        size_dim   = self._score_company_size(company, icp, ev_by_field)
        growth_dim = self._score_growth_signals(company, icp, ev_by_field)
        hiring_dim = self._score_hiring_signals(company, ev_by_field)
        auto_dim   = self._score_automation_opportunity(company, icp, ev_by_field)
        penalty    = self._compute_disqualifier_penalty(company, icp)

        raw = (icp_dim.score + ind_dim.score + size_dim.score +
               growth_dim.score + hiring_dim.score + auto_dim.score)
        total = max(0, min(100, raw - penalty))

        return LeadScore(
            company_id=company["id"],
            icp_match=icp_dim, industry_fit=ind_dim, company_size=size_dim,
            growth_signals=growth_dim, hiring_signals=hiring_dim,
            automation_opportunity=auto_dim,
            total_score=total, score_category=categorize_score(total),
            scored_at=datetime.now(timezone.utc),
        )

    def _score_icp_match(self, company, icp, ev_by_field) -> ScoreDimension:
        MAX = 25
        score = 0
        ev_ids: list[str] = []
        reasons: list[str] = []

        company_text = " ".join(
            company.get("services", []) + [company.get("industry", "")]
        ).lower()

        pain_hits = keyword_overlap(company_text, icp.get("pain_points", []))
        pain_score = min(10, len(pain_hits) * 3)
        score += pain_score
        if pain_hits:
            reasons.append(f"matches {len(pain_hits)} pain point(s): {', '.join(pain_hits[:2])}")
            ev_ids += [e["id"] for e in ev_by_field.get("service", [])[:3]]

        pos_hits = keyword_overlap(
            company_text + " " + " ".join(company.get("technologies", [])).lower(),
            icp.get("positive_signals", [])
        )
        pos_score = min(10, len(pos_hits) * 3)
        score += pos_score
        if pos_hits:
            reasons.append(f"matches {len(pos_hits)} positive signal(s)")
            ev_ids += [e["id"] for e in ev_by_field.get("technology", [])[:2]]

        target_locs = [l.lower() for l in icp.get("target_locations", [])]
        loc = company.get("location", "").lower()
        if loc and any(tl in loc for tl in target_locs):
            score += 5
            reasons.append("location matches ICP target region")
            ev_ids += [e["id"] for e in ev_by_field.get("location", [])[:1]]

        return ScoreDimension(
            score=min(MAX, score), max_score=MAX,
            reason="; ".join(reasons) or "No significant ICP signal overlap",
            evidence_ids=list(set(ev_ids)),
        )

    def _score_industry_fit(self, company, icp, ev_by_field) -> ScoreDimension:
        MAX = 20
        ci = (company.get("industry") or "").lower()
        targets = [i.lower() for i in icp.get("target_industries", [])]
        ev_ids = [e["id"] for e in ev_by_field.get("industry", [])[:1]]

        if not ci:
            return ScoreDimension(score=0, max_score=MAX,
                                   reason="Industry not detected", evidence_ids=[])
        if ci in targets:
            return ScoreDimension(score=20, max_score=MAX,
                                   reason=f"Industry '{company.get('industry')}' directly matches ICP",
                                   evidence_ids=ev_ids)
        if any(t in ci or ci in t for t in targets):
            return ScoreDimension(score=12, max_score=MAX,
                                   reason=f"Industry '{company.get('industry')}' partially matches ICP",
                                   evidence_ids=ev_ids)
        return ScoreDimension(score=0, max_score=MAX,
                               reason=f"Industry '{company.get('industry')}' does not match ICP",
                               evidence_ids=ev_ids)

    def _score_company_size(self, company, icp, ev_by_field) -> ScoreDimension:
        MAX = 15
        cr = company.get("employee_range", "")
        targets = icp.get("target_company_sizes", [])
        ev_ids = [e["id"] for e in ev_by_field.get("employee_range", [])[:1]]

        if not cr or not targets:
            return ScoreDimension(score=5, max_score=MAX,
                                   reason="Company size unknown — partial credit", evidence_ids=[])
        if cr in targets:
            return ScoreDimension(score=15, max_score=MAX,
                                   reason=f"Size '{cr}' matches ICP target", evidence_ids=ev_ids)
        try:
            ci = _SIZE_TIERS.index(cr)
            tis = [_SIZE_TIERS.index(s) for s in targets if s in _SIZE_TIERS]
            if tis:
                dist = min(abs(ci - ti) for ti in tis)
                if dist == 1:
                    return ScoreDimension(score=8, max_score=MAX,
                                           reason=f"Size '{cr}' adjacent to ICP target",
                                           evidence_ids=ev_ids)
                if dist == 2:
                    return ScoreDimension(score=3, max_score=MAX,
                                           reason=f"Size '{cr}' somewhat outside ICP target",
                                           evidence_ids=ev_ids)
        except ValueError:
            pass
        return ScoreDimension(score=0, max_score=MAX,
                               reason=f"Size '{cr}' outside ICP target range",
                               evidence_ids=ev_ids)

    def _score_growth_signals(self, company, icp, ev_by_field) -> ScoreDimension:
        MAX = 20
        score = 0
        ev_ids: list[str] = []
        reasons: list[str] = []

        if company.get("booking_detected"):
            score += 5
            reasons.append("online booking widget detected")
            ev_ids += [e["id"] for e in ev_by_field.get("booking_widget", [])[:1]]

        tech_hits = [t for t in company.get("technologies", []) if t in _GROWTH_TECH_SIGNALS]
        tech_score = min(5, len(tech_hits) * 2)
        score += tech_score
        if tech_hits:
            reasons.append(f"growth tech: {', '.join(tech_hits[:2])}")
            ev_ids += [e["id"] for e in ev_by_field.get("technology", [])[:2]]

        company_text = " ".join(
            company.get("services", []) +
            [company.get("location", ""), company.get("industry", "")]
        ).lower()
        trigger_hits = keyword_overlap(company_text, icp.get("buying_triggers", []))
        score += min(5, len(trigger_hits) * 2)
        if trigger_hits:
            reasons.append(f"matches {len(trigger_hits)} buying trigger(s)")

        if company.get("hiring_signals"):
            score += 5
            reasons.append("active hiring indicates growth")
            ev_ids += [e["id"] for e in ev_by_field.get("hiring_signal", [])[:1]]

        return ScoreDimension(
            score=min(MAX, score), max_score=MAX,
            reason="; ".join(reasons) or "No significant growth signals",
            evidence_ids=list(set(ev_ids)),
        )

    def _score_hiring_signals(self, company, ev_by_field) -> ScoreDimension:
        MAX = 10
        ev_ids = [e["id"] for e in ev_by_field.get("hiring_signal", [])[:1]]
        if not company.get("hiring_signals"):
            return ScoreDimension(score=0, max_score=MAX,
                                   reason="No active hiring signals found", evidence_ids=[])
        titles = company.get("job_titles_found", [])
        if titles:
            preview = ", ".join(titles[:3]) + ("..." if len(titles) > 3 else "")
            return ScoreDimension(score=10, max_score=MAX,
                                   reason=f"Actively hiring: {preview}", evidence_ids=ev_ids)
        return ScoreDimension(score=6, max_score=MAX,
                               reason="Hiring language detected, no specific roles identified",
                               evidence_ids=ev_ids)

    def _score_automation_opportunity(self, company, icp, ev_by_field) -> ScoreDimension:
        MAX = 10
        score = 0
        reasons: list[str] = []

        text = " ".join(company.get("services", []) +
                        [company.get("industry", "")]).lower()
        manual_hits = keyword_overlap(text, _AUTOMATION_KEYWORDS)
        manual_score = min(6, len(manual_hits) * 2)
        score += manual_score
        if manual_hits:
            reasons.append(f"manual process signals: {', '.join(manual_hits[:2])}")

        service_industries = {"dental", "healthcare", "legal", "accounting",
                               "consulting", "fitness", "beauty", "medical"}
        if any(kw in text for kw in service_industries) and not company.get("booking_detected"):
            score += 3
            reasons.append("service business without online booking")

        if not company.get("technologies"):
            score += 1
            reasons.append("no technology signals detected")

        return ScoreDimension(
            score=min(MAX, score), max_score=MAX,
            reason="; ".join(reasons) or "No clear automation opportunity signals",
            evidence_ids=[],
        )

    def _compute_disqualifier_penalty(self, company, icp) -> int:
        disqualifiers = icp.get("disqualifying_signals", [])
        if not disqualifiers:
            return 0
        text = " ".join(company.get("services", []) +
                        company.get("technologies", []) +
                        [company.get("industry", "")]).lower()
        hits = keyword_overlap(text, disqualifiers)
        return min(15, len(hits) * 5)

    def _persist_score(self, score: LeadScore) -> None:
        breakdown = {
            dim: {
                "score": getattr(score, dim).score,
                "max_score": getattr(score, dim).max_score,
                "reason": getattr(score, dim).reason,
                "evidence_ids": getattr(score, dim).evidence_ids,
            }
            for dim in ["icp_match", "industry_fit", "company_size",
                        "growth_signals", "hiring_signals", "automation_opportunity"]
        }
        all_ev_ids = list(set(
            score.icp_match.evidence_ids + score.industry_fit.evidence_ids +
            score.company_size.evidence_ids + score.growth_signals.evidence_ids +
            score.hiring_signals.evidence_ids + score.automation_opportunity.evidence_ids
        ))
        try:
            self._db.table("lead_scores").upsert({
                "id": str(uuid.uuid4()),
                "company_id": score.company_id,
                "icp_match": score.icp_match.score,
                "industry_fit": score.industry_fit.score,
                "company_size": score.company_size.score,
                "growth_signals": score.growth_signals.score,
                "hiring_signals": score.hiring_signals.score,
                "automation_opportunity": score.automation_opportunity.score,
                "total_score": score.total_score,
                "score_category": score.score_category,
                "score_breakdown": breakdown,
                "evidence_used": all_ev_ids,
                "scored_at": score.scored_at.isoformat(),
            }, on_conflict="company_id").execute()
        except Exception as exc:
            logger.error("persist_score_failed", company_id=score.company_id, error=str(exc))
            raise

    def _load_company(self, company_id: str) -> dict:
        result = self._db.table("companies").select("*").eq("id", company_id).single().execute()
        if not result.data:
            raise ValueError(f"Company {company_id} not found")
        return result.data

    def _load_icp(self, icp_id: str) -> dict:
        result = (
            self._db.table("icps")
            .select("target_industries, target_company_sizes, target_locations, "
                    "pain_points, buying_triggers, positive_signals, disqualifying_signals")
            .eq("id", icp_id).single().execute()
        )
        if not result.data:
            raise ValueError(f"ICP {icp_id} not found")
        return result.data

    def _load_evidence(self, company_id: str) -> list[dict]:
        result = (
            self._db.table("evidence")
            .select("id, field_name, value, confidence, trust_level, source_url")
            .eq("company_id", company_id).lte("trust_level", 2).execute()
        )
        return result.data or []

    def _load_complete_companies(self, search_id: str) -> list[dict]:
        result = (
            self._db.table("companies")
            .select("id, domain, industry, employee_range, services, technologies, "
                    "hiring_signals, job_titles_found, location, booking_detected, "
                    "data_confidence, research_status")
            .eq("search_id", search_id).eq("research_status", "complete").execute()
        )
        return result.data or []

    def _already_scored(self, company_id: str) -> bool:
        try:
            result = (self._db.table("lead_scores").select("id")
                      .eq("company_id", company_id).limit(1).execute())
            return bool(result.data)
        except Exception:
            return False
