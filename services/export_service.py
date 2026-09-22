"""
services/export_service.py
────────────────────────────
Deterministic CSV export. Zero Groq calls.

Builds one CSV row per researched company for a search_id, left-joining
in that company's lead_score, qualification, and outreach_messages if
they exist (blank columns if a company hasn't reached that pipeline
stage yet — export is meant to work at any point in the pipeline, not
only after every company has been fully qualified and pitched).

Follows the same read-only loading pattern as ScoringService/
QualifyService/OutreachService: fetch companies for search_id, collect
their IDs, then fetch related rows with `.in_("company_id", ids)` —
the Supabase client's fluent API doesn't support cross-table joins,
so this mirrors QualifyService._load_scored_companies rather than
introducing a new pattern.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from supabase import Client

from core.errors import NotFoundError
from core.logging import get_logger

logger = get_logger(__name__)


# Column order for the exported CSV. Score dimension columns are
# prefixed with "score_" to avoid colliding with the company table's
# own boolean "hiring_signals" column.
_CSV_COLUMNS = [
    # Company (companies table)
    "company_name",
    "domain",
    "website",
    "location",
    "industry",
    "employee_range",
    "services",
    "technologies",
    "hiring_signals",
    "job_titles_found",
    "contact_emails",
    "contact_phones",
    "contact_pages",
    "linkedin_url",
    "booking_detected",
    "research_status",
    "data_confidence",
    # Score (lead_scores table)
    "score_total",
    "score_category",
    "score_icp_match",
    "score_industry_fit",
    "score_company_size",
    "score_growth_signals",
    "score_hiring_signals",
    "score_automation_opportunity",
    "score_disqualifiers",
    # Qualification (qualifications table)
    "qualification_status",
    "qualification_confidence",
    "why_qualified",
    "pain_points",
    "suggested_services",
    "outreach_angle",
    "buying_triggers",
    # Outreach (outreach_messages table)
    "linkedin_message",
    "cold_email_subject",
    "cold_email_body",
    "audit_opener",
    "followup_message",
]


class ExportService:
    def __init__(self, db: Client) -> None:
        self._db = db

    # ── Ownership ─────────────────────────────────────────────────────────

    def verify_ownership(self, search_id: str, fingerprint: str) -> None:
        """
        Raise NotFoundError if the search doesn't exist or doesn't belong
        to this fingerprint. Same NotFoundError either way — never reveal
        whether a search_id exists but belongs to someone else.
        """
        try:
            result = (
                self._db.table("searches")
                .select("fingerprint")
                .eq("id", search_id)
                .single()
                .execute()
            )
            row = result.data
        except Exception:
            row = None

        if not row or row.get("fingerprint") != fingerprint:
            raise NotFoundError("Search", search_id)

    # ── CSV Build ─────────────────────────────────────────────────────────

    def build_csv(self, search_id: str) -> str:
        """
        Build the full CSV export for a search_id as a string.

        Rows are sorted by score_total descending (companies with no
        score yet sort last), matching the "highest-first" convention
        already used by ScoringService.score_batch.
        """
        companies = self._load_companies(search_id)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        if not companies:
            return buffer.getvalue()

        company_ids = [c["id"] for c in companies]
        scores = self._load_by_company_ids("lead_scores", company_ids)
        qualifications = self._load_by_company_ids("qualifications", company_ids)
        outreach = self._load_by_company_ids("outreach_messages", company_ids)

        rows = [
            self._flatten(
                company,
                scores.get(company["id"]),
                qualifications.get(company["id"]),
                outreach.get(company["id"]),
            )
            for company in companies
        ]
        rows.sort(
            key=lambda r: r["score_total"] if r["score_total"] != "" else -1,
            reverse=True,
        )

        writer.writerows(rows)
        return buffer.getvalue()

    # ── Loaders (read-only) ──────────────────────────────────────────────

    def _load_companies(self, search_id: str) -> list[dict]:
        result = (
            self._db.table("companies")
            .select("*")
            .eq("search_id", search_id)
            .execute()
        )
        return result.data or []

    def _load_by_company_ids(self, table: str, company_ids: list[str]) -> dict[str, dict]:
        if not company_ids:
            return {}
        try:
            result = self._db.table(table).select("*").in_("company_id", company_ids).execute()
        except Exception as exc:
            logger.warning("export_load_failed", table=table, error=str(exc))
            return {}
        return {row["company_id"]: row for row in (result.data or [])}

    # ── Flatten ───────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(
        company: dict,
        score: dict | None,
        qualification: dict | None,
        outreach: dict | None,
    ) -> dict[str, Any]:
        social_profiles = company.get("social_profiles") or {}

        row: dict[str, Any] = {
            "company_name": company.get("name", ""),
            "domain": company.get("domain", ""),
            "website": company.get("website", ""),
            "location": company.get("location", ""),
            "industry": company.get("industry", ""),
            "employee_range": company.get("employee_range", ""),
            "services": _join(company.get("services")),
            "technologies": _join(company.get("technologies")),
            "hiring_signals": company.get("hiring_signals", False),
            "job_titles_found": _join(company.get("job_titles_found")),
            "contact_emails": _join(company.get("contact_emails")),
            "contact_phones": _join(company.get("contact_phones")),
            "contact_pages": _join(company.get("contact_pages")),
            "linkedin_url": social_profiles.get("linkedin", ""),
            "booking_detected": company.get("booking_detected", False),
            "research_status": company.get("research_status", ""),
            "data_confidence": company.get("data_confidence", ""),
            "score_total": "",
            "score_category": "",
            "score_icp_match": "",
            "score_industry_fit": "",
            "score_company_size": "",
            "score_growth_signals": "",
            "score_hiring_signals": "",
            "score_automation_opportunity": "",
            "score_disqualifiers": "",
            "qualification_status": "",
            "qualification_confidence": "",
            "why_qualified": "",
            "pain_points": "",
            "suggested_services": "",
            "outreach_angle": "",
            "buying_triggers": "",
            "linkedin_message": "",
            "cold_email_subject": "",
            "cold_email_body": "",
            "audit_opener": "",
            "followup_message": "",
        }

        if score:
            row.update({
                "score_total": score.get("total_score", ""),
                "score_category": score.get("score_category", ""),
                "score_icp_match": score.get("icp_match", ""),
                "score_industry_fit": score.get("industry_fit", ""),
                "score_company_size": score.get("company_size", ""),
                "score_growth_signals": score.get("growth_signals", ""),
                "score_hiring_signals": score.get("hiring_signals", ""),
                "score_automation_opportunity": score.get("automation_opportunity", ""),
                "score_disqualifiers": score.get("score_breakdown", {}).get("disqualifiers", {}).get("score", ""),
            })

        if qualification:
            row.update({
                "qualification_status": qualification.get("qualification_status", ""),
                "qualification_confidence": qualification.get("confidence", ""),
                "why_qualified": qualification.get("why_qualified", ""),
                "pain_points": _join(qualification.get("pain_points")),
                "suggested_services": _join(qualification.get("suggested_services")),
                "outreach_angle": qualification.get("outreach_angle", ""),
                "buying_triggers": _join(qualification.get("buying_triggers")),
            })

        if outreach:
            row.update({
                "linkedin_message": outreach.get("linkedin_message", ""),
                "cold_email_subject": outreach.get("cold_email_subject", ""),
                "cold_email_body": outreach.get("cold_email_body", ""),
                "audit_opener": outreach.get("audit_opener", ""),
                "followup_message": outreach.get("followup_message", ""),
            })

        return row


def _join(values: list[str] | None) -> str:
    if not values:
        return ""
    return "; ".join(str(v) for v in values)