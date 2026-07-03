"""
crawlers/extractors/contact.py
────────────────────────────────
Extract contact information from crawled pages.

Extracts:
  - Email addresses (F_CONTACT_EMAIL) — regex + mailto: links
  - Phone numbers (F_CONTACT_PHONE) — regex + tel: links, normalized
  - Contact page URLs (F_CONTACT_PAGE) — identified /contact pages

Phone numbers ARE extracted (purely factual data — useful for Phase 6
outreach and as a completeness signal for the research object). They are
normalized to a consistent digits-and-plus format so Phase 5/6 never has
to deal with inconsistent formatting like "(555) 123-4567" vs
"555.123.4567" vs "+1 555 123 4567" representing the same number.

Email filtering:
  - Skips obvious no-reply, noreply, info@ generic addresses
    (kept if they're the only address, since they're still useful)
  - Skips image/asset emails that appear in src attributes
  - Deduplicates across pages

Phone filtering:
  - Requires at least 10 digits (rejects partial/malformed matches)
  - Rejects sequences that are obviously not phone numbers (all same
    digit, sequential digits) — common false-positive sources in
    minified JS or tracking pixel query strings
  - Deduplicates by normalized digit sequence, not raw string
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    ExtractedFact,
    F_CONTACT_EMAIL,
    F_CONTACT_PAGE,
    F_CONTACT_PHONE,
    source_type_for_url,
    SOURCE_CONTACT,
)
from crawlers.p1_crawler import PageResult

_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

# Domains that are never real contact emails
_SKIP_EMAIL_DOMAINS = frozenset({
    "example.com", "test.com", "domain.com", "email.com",
    "sentry.io", "wixpress.com", "squarespace.com",
})

# Prefixes that suggest a generic/automated address
_GENERIC_PREFIXES = frozenset({
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "notifications", "bounce", "mailer-daemon", "postmaster",
    "webmaster", "abuse", "spam",
})

_MAX_EMAILS = 5
_MAX_PHONES = 3

# Phone pattern: matches common US/international formats.
# Requires a leading boundary so it doesn't match mid-string digit runs
# (e.g. order IDs, tracking numbers embedded in scripts).
_PHONE_RE = re.compile(
    r"(?<![\d/])"
    r"(\+?\d{1,3}[-.\s]?)?"
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    r"(?!\d)"
)

# Sequences that are almost certainly not real phone numbers
_REJECTED_DIGIT_PATTERNS = (
    re.compile(r"^(\d)\1{9,}$"),          # all same digit, e.g. 0000000000
    re.compile(r"^0123456789$"),           # sequential
    re.compile(r"^1234567890$"),           # sequential
)


def _is_useful_email(email: str) -> bool:
    """Return True if the email is worth storing."""
    email = email.lower()
    local, _, domain = email.rpartition("@")
    if domain in _SKIP_EMAIL_DOMAINS:
        return False
    if local in _GENERIC_PREFIXES:
        return False
    # Skip image/asset URLs that contain @ (rare but happens with data URIs)
    if len(email) > 120:
        return False
    return True


def _normalize_phone(raw: str) -> str | None:
    """
    Normalize a raw phone match to a consistent digits-and-plus format.

    "+1 (555) 123-4567" -> "+15551234567"
    "(555) 123-4567"     -> "5551234567"
    "555.123.4567"       -> "5551234567"

    Returns None if the result doesn't look like a plausible phone number.
    """
    has_plus = raw.strip().startswith("+")
    digits = re.sub(r"\D", "", raw)

    if len(digits) < 10 or len(digits) > 15:
        return None

    for pattern in _REJECTED_DIGIT_PATTERNS:
        if pattern.match(digits[-10:]):
            return None

    return f"+{digits}" if has_plus else digits


def extract_contact(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Extract contact emails, phone numbers, and contact page URLs.
    """
    facts: list[ExtractedFact] = []
    seen_emails: set[str] = set()
    seen_phones: set[str] = set()
    contact_page_stored = False

    for page in pages:
        if not page.ok:
            continue

        source_type = source_type_for_url(page.url)
        is_contact = source_type == SOURCE_CONTACT

        # Record the contact page URL itself
        if is_contact and not contact_page_stored:
            contact_page_stored = True
            facts.append(ExtractedFact(
                field_name=F_CONTACT_PAGE,
                value=page.url,
                source_url=page.url,
                source_type=source_type,
                confidence=0.95,
            ))

        try:
            soup = BeautifulSoup(page.html, "lxml")

            # ── Emails: mailto: links first (highest confidence) ──────────
            for a_tag in soup.find_all("a", href=re.compile(r"^mailto:", re.I)):
                href = a_tag["href"].replace("mailto:", "").split("?")[0].strip().lower()
                if href and href not in seen_emails and _is_useful_email(href):
                    seen_emails.add(href)
                    facts.append(ExtractedFact(
                        field_name=F_CONTACT_EMAIL,
                        value=href,
                        source_url=page.url,
                        source_type=source_type,
                        confidence=0.95 if is_contact else 0.85,
                    ))

            # ── Phones: tel: links first (highest confidence) ─────────────
            for a_tag in soup.find_all("a", href=re.compile(r"^tel:", re.I)):
                href = a_tag["href"].replace("tel:", "").strip()
                normalized = _normalize_phone(href)
                if normalized and normalized not in seen_phones and len(seen_phones) < _MAX_PHONES:
                    seen_phones.add(normalized)
                    facts.append(ExtractedFact(
                        field_name=F_CONTACT_PHONE,
                        value=normalized,
                        source_url=page.url,
                        source_type=source_type,
                        confidence=0.95 if is_contact else 0.85,
                    ))

            # ── Regex scan on visible text ──────────────────────────────
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)

            for match in _EMAIL_RE.finditer(text):
                email = match.group(0).lower()
                if email not in seen_emails and _is_useful_email(email):
                    seen_emails.add(email)
                    facts.append(ExtractedFact(
                        field_name=F_CONTACT_EMAIL,
                        value=email,
                        source_url=page.url,
                        source_type=source_type,
                        confidence=0.80 if is_contact else 0.70,
                    ))

            if len(seen_phones) < _MAX_PHONES:
                for match in _PHONE_RE.finditer(text):
                    normalized = _normalize_phone(match.group(0))
                    if normalized and normalized not in seen_phones:
                        seen_phones.add(normalized)
                        facts.append(ExtractedFact(
                            field_name=F_CONTACT_PHONE,
                            value=normalized,
                            source_url=page.url,
                            source_type=source_type,
                            confidence=0.70 if is_contact else 0.55,
                        ))
                    if len(seen_phones) >= _MAX_PHONES:
                        break

            if len(seen_emails) >= _MAX_EMAILS and len(seen_phones) >= _MAX_PHONES:
                break

        except Exception:
            continue

    return facts
