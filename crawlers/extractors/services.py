"""
crawlers/extractors/services.py
────────────────────────────────
Extract service/product offerings from crawled pages.

Approach: heading-first extraction. Services are almost always listed
under <h2>/<h3> headings on the homepage or /services page, or in
navigation items. This is deterministic — no LLM.

Returns one ExtractedFact per distinct service found (field_name=F_SERVICE).
Phase 5 scoring reads the full list; having multiple facts is correct.

Confidence levels:
  0.85 — from /services page headings (most authoritative)
  0.75 — from homepage headings
  0.65 — from navigation items
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    ExtractedFact,
    F_SERVICE,
    SOURCE_HOMEPAGE,
    SOURCE_SERVICES,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult

# Maximum services to extract (avoid noise from content-heavy pages)
_MAX_SERVICES = 15

# Minimum and maximum word count for a plausible service name
_MIN_WORDS = 2
_MAX_WORDS = 7

# Generic headings/labels that are NOT service names.
# Comprehensive list covering navigation labels, UI chrome, legal pages,
# and common non-service section headings.
_SKIP_HEADINGS = frozenset({
    # Navigation / structure
    "home", "homepage", "menu", "navigation", "nav", "header", "footer",
    "sidebar", "breadcrumb", "search", "sitemap",
    # About / company
    "about", "about us", "our story", "who we are", "our history",
    "meet the team", "our team", "team", "leadership", "management",
    "mission", "our mission", "vision", "our vision", "values", "our values",
    "culture", "careers", "jobs", "join us", "join our team", "work with us",
    "work here", "we're hiring", "we are hiring",
    # Contact / booking
    "contact", "contact us", "get in touch", "reach us", "reach out",
    "schedule", "book", "book now", "book an appointment",
    "appointment", "appointments", "new patients", "existing patients",
    "patient portal", "patient login",
    # CMS / blog
    "blog", "news", "press", "media", "articles", "posts", "updates",
    "announcements", "events", "resources", "resource center", "library",
    "guides", "ebooks", "downloads", "case studies",
    # Legal / policy
    "privacy", "privacy policy", "terms", "terms of service",
    "terms and conditions", "cookie policy", "legal", "disclaimer",
    "accessibility", "compliance",
    # Auth / account
    "login", "log in", "sign in", "signup", "sign up", "register",
    "create account", "my account", "dashboard", "portal",
    # CTA / UI labels
    "read more", "learn more", "find out more", "see more", "view all",
    "click here", "get started", "start now", "try now", "try free",
    "submit", "send", "send message", "send us a message",
    "next", "previous", "back", "continue", "go", "more",
    "explore", "discover",
    # Generic section headers
    "services", "our services", "what we do", "solutions", "our solutions",
    "offerings", "overview", "introduction", "welcome",
    "why choose us", "why us", "how it works", "how we work",
    "get a quote", "free quote", "free consultation", "free trial",
    "testimonials", "reviews", "what our clients say", "what people say",
    "gallery", "portfolio", "case studies", "success stories",
    "partners", "clients", "our clients", "our partners",
    "awards", "certifications", "accreditations",
    "faqs", "faq", "frequently asked questions", "questions", "q&a",
    "pricing", "plans", "packages", "packages and pricing",
    "features", "benefits",
})

# Patterns that indicate a CTA, label, or non-service heading
_SKIP_PATTERNS = re.compile(
    r"^(contact|call|email|book|schedule|request|get|find|discover|"
    r"learn|see|view|explore|read|click|start|try|sign|log|download|"
    r"submit|send|go to|back to|return to|skip|next|previous|more)\b",
    re.IGNORECASE,
)

# Headings that are purely numeric, years, or punctuation
_NUMERIC_OR_YEAR = re.compile(r"^[\d\s\W]+$")
_YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")


def _normalize_key(text: str) -> str:
    """Produce a case-insensitive, whitespace-normalised dedup key."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _clean_service_text(text: str) -> str | None:
    """
    Clean heading text and return if it looks like a real service name.

    Returns None if the text should be rejected.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Strip trailing punctuation (but not hyphens that may be part of a name)
    text = re.sub(r"[.:,;!?]+$", "", text).strip()

    if not text:
        return None

    # Reject if only whitespace/punctuation/numbers remain after strip
    if _NUMERIC_OR_YEAR.match(text):
        return None

    # Reject standalone years
    if _YEAR_PATTERN.match(text.strip()):
        return None

    words = text.split()
    if len(words) < _MIN_WORDS or len(words) > _MAX_WORDS:
        return None

    # Case-insensitive skip list check
    if _normalize_key(text) in _SKIP_HEADINGS:
        return None

    # CTA / UI label patterns
    if _SKIP_PATTERNS.match(text):
        return None

    # Must start with a letter (not a symbol or number-only prefix)
    if not text[0].isalpha():
        return None

    return text


def _extract_from_headings(
    soup: BeautifulSoup, source_url: str, confidence: float
) -> list[ExtractedFact]:
    """Extract service names from h2/h3 headings."""
    facts: list[ExtractedFact] = []
    seen: set[str] = set()
    source_type = source_type_for_url(source_url)

    for tag_name in ("h2", "h3"):
        for heading in soup.find_all(tag_name):
            text = heading.get_text(separator=" ", strip=True)
            cleaned = _clean_service_text(text)
            if not cleaned:
                continue
            key = _normalize_key(cleaned)
            if key not in seen:
                seen.add(key)
                facts.append(ExtractedFact(
                    field_name=F_SERVICE,
                    value=cleaned,
                    source_url=source_url,
                    source_type=source_type,
                    confidence=confidence,
                ))
            if len(facts) >= _MAX_SERVICES:
                return facts

    return facts


def _extract_from_nav(
    soup: BeautifulSoup, source_url: str
) -> list[ExtractedFact]:
    """Extract services listed in navigation menus."""
    facts: list[ExtractedFact] = []
    seen: set[str] = set()
    source_type = source_type_for_url(source_url)

    nav_elements = soup.find_all(
        ["nav", "ul"],
        attrs={"class": re.compile(r"nav|menu|header", re.I)},
    )

    for nav in nav_elements[:3]:
        for link in nav.find_all("a"):
            text = link.get_text(separator=" ", strip=True)
            cleaned = _clean_service_text(text)
            if not cleaned:
                continue
            key = _normalize_key(cleaned)
            if key not in seen:
                seen.add(key)
                facts.append(ExtractedFact(
                    field_name=F_SERVICE,
                    value=cleaned,
                    source_url=source_url,
                    source_type=source_type,
                    confidence=0.65,
                ))

    return facts[:10]


def extract_services(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Extract service/offering facts from crawled pages.

    Services page headings are the most authoritative source.
    Deduplication is case-insensitive and whitespace-normalised so
    "Dental Implants", "dental implants", and "Dental implants" all
    resolve to a single entry.

    Returns one fact per distinct service found.
    """
    all_facts: list[ExtractedFact] = []
    seen: set[str] = set()  # normalised keys — case-insensitive global dedup

    for page in pages:
        if not page.ok:
            continue
        try:
            soup = BeautifulSoup(page.html, "lxml")
        except Exception:
            continue

        source_type = source_type_for_url(page.url)

        # Confidence depends on which page we're on
        if source_type == SOURCE_SERVICES:
            confidence = 0.85
        elif source_type == SOURCE_HOMEPAGE:
            confidence = 0.75
        else:
            confidence = 0.70

        heading_facts = _extract_from_headings(soup, page.url, confidence)
        nav_facts = _extract_from_nav(soup, page.url)

        for fact in heading_facts + nav_facts:
            key = _normalize_key(fact.value)
            if key not in seen:
                seen.add(key)
                all_facts.append(fact)

        if len(all_facts) >= _MAX_SERVICES:
            break

    return all_facts[:_MAX_SERVICES]
