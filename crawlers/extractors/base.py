"""
crawlers/extractors/base.py
────────────────────────────
Shared types returned by every extractor.

Every extractor returns list[ExtractedFact]. The research service
collects these, writes them to the evidence table, then derives the
company row from them. Nothing outside this package ever reads raw HTML.

field_name constants are defined here so extractors stay consistent
and Phase 5 scoring can reference them without string literals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractedFact:
    """
    A single verifiable fact extracted from a crawled page.

    Maps 1-to-1 with a row in the evidence table.
    """

    field_name: str       # one of the F_* constants below
    value: str            # always a string — lists are stored as separate facts
    source_url: str       # the exact URL this was extracted from
    source_type: str      # one of the SOURCE_* constants below
    confidence: float     # 0.0–1.0
    trust_level: int = 1  # 1 for P1 (company site), always 1 in this crawler


# ── field_name constants ──────────────────────────────────────────────────────

F_NAME              = "name"
F_INDUSTRY          = "industry"
F_LOCATION          = "location"
F_EMPLOYEE_RANGE    = "employee_range"
F_SERVICE           = "service"
F_TECHNOLOGY        = "technology"
F_HIRING_SIGNAL     = "hiring_signal"
F_JOB_TITLE         = "job_title"
F_CONTACT_EMAIL     = "contact_email"
F_CONTACT_PHONE      = "contact_phone"
F_CONTACT_PAGE      = "contact_page"
F_SOCIAL_LINKEDIN   = "social_linkedin"
F_SOCIAL_TWITTER    = "social_twitter"
F_SOCIAL_FACEBOOK   = "social_facebook"
F_SOCIAL_INSTAGRAM  = "social_instagram"
F_BOOKING_WIDGET    = "booking_widget"

# ── source_type constants ─────────────────────────────────────────────────────

SOURCE_HOMEPAGE     = "company_homepage"
SOURCE_ABOUT        = "company_about"
SOURCE_SERVICES     = "company_services"
SOURCE_TEAM         = "company_team"
SOURCE_CAREERS      = "company_careers"
SOURCE_CONTACT      = "company_contact"


def source_type_for_url(url: str) -> str:
    """
    Infer a source_type string from a crawled URL path.
    Falls back to SOURCE_HOMEPAGE for unrecognised paths.
    """
    path = url.lower().rstrip("/")
    if any(s in path for s in ("/about", "/who-we-are", "/our-story")):
        return SOURCE_ABOUT
    if any(s in path for s in ("/service", "/what-we-do", "/solution", "/offering")):
        return SOURCE_SERVICES
    if any(s in path for s in ("/team", "/leadership", "/people")):
        return SOURCE_TEAM
    if any(s in path for s in ("/career", "/job", "/join", "/work-with")):
        return SOURCE_CAREERS
    if any(s in path for s in ("/contact", "/get-in-touch", "/reach")):
        return SOURCE_CONTACT
    return SOURCE_HOMEPAGE
