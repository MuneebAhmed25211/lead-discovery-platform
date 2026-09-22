"""
crawlers/extractors/employee_count.py
──────────────────────────────────────
Extract employee count / company size signals from crawled pages.

Approach: regex pattern matching against visible text. Companies often
mention their team size in about pages, press releases, or footers.

Returns a normalized range string matching the ICP schema:
  "1-10" | "11-50" | "51-200" | "201-500" | "501-1000" | "1000+"

Returns at most one fact — the highest (page-weighted) confidence
size signal found.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    ExtractedFact,
    F_EMPLOYEE_RANGE,
    SOURCE_ABOUT,
    SOURCE_CAREERS,
    SOURCE_HOMEPAGE,
    SOURCE_TEAM,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult


def _normalize_count(n: int) -> str:
    """Map a raw employee count to the canonical range string."""
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 200:
        return "51-200"
    if n <= 500:
        return "201-500"
    if n <= 1000:
        return "501-1000"
    return "1000+"


# ── Page-type confidence weighting ──────────────────────────────────────────
#
# Spec priority: About page > Team page > Careers > Homepage > other.
# A headcount stated on an About/Team page is describing the company
# itself; the same phrase on a homepage is more often marketing copy
# ("join our team of 500 happy customers") and on a careers/services page
# it's more likely to be about a specific role than the whole company.
# Weights are multiplicative against each pattern's base confidence so
# page type and phrase specificity combine into a single comparable score.
_PAGE_TYPE_WEIGHT: dict[str, float] = {
    SOURCE_ABOUT: 1.00,
    SOURCE_TEAM: 0.97,
    SOURCE_CAREERS: 0.90,
    SOURCE_HOMEPAGE: 0.82,
}
_DEFAULT_PAGE_WEIGHT = 0.75  # services/contact pages — rarely state headcount


def _page_weight(source_type: str) -> float:
    return _PAGE_TYPE_WEIGHT.get(source_type, _DEFAULT_PAGE_WEIGHT)


# ── Number fragment ──────────────────────────────────────────────────────────
#
# Matches plain integers ("45"), comma-grouped thousands ("2,500"), and an
# optional trailing "+" ("35+", "250+"). The "+" is intentionally matched
# but NOT captured — the captured group is always clean digits/commas so
# downstream int() parsing needs no extra stripping.
_NUM = r"(\d{1,3}(?:,\d{3})+|\d{1,6})\+?"

# Guard against truncating a range expression down to just its first
# number. Patterns below anchor on a keyword ("team of", "we employ", ...)
# without requiring a specific noun immediately after the number, which
# means "team of 1 to 10 developers" — extremely common staff-augmentation
# marketing copy on dev/AI agency sites — was matching "team of 1" and
# reporting a false "1-10" headcount for companies that are not remotely
# that size. If the number is immediately followed by "to N" or "-N", it's
# part of a range this pattern isn't equipped to parse correctly (the noun
# that would let _RANGE_PATTERN handle it, e.g. "developers", is
# deliberately not in _RANGE_PATTERN's noun list either — see the
# professional-noun pattern's comment below) — so decline to match at all
# rather than silently picking the wrong number.
_NO_RANGE_TAIL = r"(?!\s*(?:to|-)\s*\d)"

# Patterns that express employee count. Every pattern is anchored to a
# keyword ("employees", "staff", "team", a job-title word, etc.) so bare
# numbers — years, phone numbers, street addresses, zip codes — never
# match; those never appear directly adjacent to headcount vocabulary.
_PATTERNS: list[tuple[re.Pattern, float]] = [
    # "team of 45" / "team of over 100" / "team of 35+"
    (re.compile(
        rf"\bteam\s+of\s+(?:over\s+|more\s+than\s+|approximately\s+|about\s+)?"
        rf"{_NUM}{_NO_RANGE_TAIL}\b",
        re.I,
    ), 0.80),
    # "our team has 35 people" / "team has grown to 40 employees"
    (re.compile(
        rf"\bteam\s+(?:has|have)\s+(?:over\s+|more\s+than\s+)?{_NUM}{_NO_RANGE_TAIL}\s+"
        rf"(?:people|employees?|staff|team\s+members?)\b",
        re.I,
    ), 0.80),
    # "45 employees" / "over 200 employees" / "2,500 employees" / "35+ employees"
    # "50,000 Associates" — confirmed real case: CHRISTUS Health (a 30,000-
    # 50,000 employee Catholic nonprofit hospital system) surfaced at
    # "MODERATE FIT" with "company size unknown" under a small-independent-
    # -practice ICP. Its own materials consistently say "50,000 Associates,"
    # never "employees" — a specific nonprofit/faith-based healthcare HR
    # convention (Ascension, Trinity Health, Providence, and others all use
    # the same "Associates" term for staff), so the size was never actually
    # unknown, just phrased in a synonym this pattern didn't recognize yet.
    (re.compile(
        rf"\b(?:over\s+|more\s+than\s+|approximately\s+|about\s+)?{_NUM}\s+"
        rf"(?:full[- ]?time\s+)?(?:employees?|associates?)\b",
        re.I,
    ), 0.85),
    # "we employ 45 people" / "a company of 200 people" / "workforce of 45"
    (re.compile(
        rf"\b(?:we\s+employ|company\s+of|workforce\s+of)\s+"
        rf"(?:over\s+|more\s+than\s+)?{_NUM}{_NO_RANGE_TAIL}\s*(?:people)?\b",
        re.I,
    ), 0.75),
    # "45 staff" / "staff of 200"
    (re.compile(rf"\b(?:staff\s+of\s+|our\s+staff\s+of\s+)?{_NUM}\s+staff\b", re.I), 0.75),
    (re.compile(rf"\bstaff\s+of\s+(?:over\s+)?{_NUM}{_NO_RANGE_TAIL}\b", re.I), 0.75),
    # "45 professionals" / "team of 45 professionals"
    (re.compile(rf"\b{_NUM}\s+(?:dedicated\s+|skilled\s+|experienced\s+)?professionals?\b", re.I), 0.70),
    # "45 dentists" / "45 physicians" — professional-practice headcount
    # nouns. "developers"/"engineers" deliberately excluded: on software/
    # AI agency sites (exactly the sites most likely to need accurate
    # headcount for correct scoring) these nouns overwhelmingly appear in
    # staff-augmentation marketing copy ("hire 1-10 developers") describing
    # a client's potential hire, not the agency's own headcount, and were
    # the single largest source of false employee_range extractions.
    (re.compile(
        rf"\b{_NUM}\s+(?:dentists?|physicians?|attorneys?|lawyers?|doctors?|consultants?)\b",
        re.I,
    ), 0.70),
    # "45+ locations" / "100+ offices" — implies multi-location = larger company.
    # Handled separately below via _normalize_from_location_count rather than
    # folded into _PATTERNS/_normalize_count: an office/branch count is not
    # an employee count, and reusing _normalize_count(n) directly mapped
    # "10 offices" straight to the "1-10" employee bucket — self-contradictory,
    # since a company with 10 physical offices necessarily has far more than
    # 10 total staff. This was the largest source of false "1-10" results on
    # real enterprises (e.g. a regional sub-page mentioning a handful of
    # offices in one country, for a company with tens of thousands of staff
    # globally) whenever no direct headcount was stated on any crawled page
    # to outrank it.
]

# Stock-ticker / publicly-traded self-description — a strong, essentially
# unambiguous signal that a crawled site is a large public company, not a
# small business. Confirmed real case: Zillow surfaced as a 33/100 "LOW
# FIT" lead under a "small businesses, 1-200 employees" Android-app-
# development ICP — its own About page (crawled via the existing /about,
# /about-us paths) self-describes as "Zillow Group, Inc. (NASDAQ: Z)
# (NASDAQ: ZG)", exactly the kind of unambiguous marker this pattern
# catches. Mapped straight to "1000+" (the existing company_size scoring
# dimension already correctly zeroes this out against any ICP that
# doesn't target "1000+"/"501-1000", so no separate gating logic is
# needed here — this is just another employee_range signal).
#
# Deliberately requires the SINGULAR "Fortune 500/100 company" — "Fortune
# 500 companies" (plural) is extremely common vendor-agency marketing
# copy ("trusted by Fortune 500 companies") describing their CLIENTS, not
# themselves, and would be a false positive if matched.
_PUBLIC_COMPANY_PATTERN = re.compile(
    r"\((?:NASDAQ|NYSE|LSE|TSX)\s*:\s*[A-Z]{1,5}\)"
    r"|\bNASDAQ\s*:\s*[A-Z]{1,5}\b"
    r"|\bNYSE\s*:\s*[A-Z]{1,5}\b"
    r"|\bpublicly[- ]traded\s+(?:company|corporation)\b"
    r"|\bFortune\s+(?:100|500)\s+company\b",
    re.I,
)

_LOCATION_PATTERN = re.compile(
    rf"\b{_NUM}\s+(?:locations?|offices?|clinics?|practices?|branches?)\b", re.I
)


def _normalize_from_location_count(n: int) -> str:
    """
    Map an office/branch/location count to a *conservative floor* on
    employee_range — never the same tier a direct headcount of N would
    imply. Each additional office implies staff well beyond just N people
    (reception, support, and service staff per site), so even a "small"
    office count like 2-10 rules out the "1-10" bucket entirely.
    """
    if n <= 1:
        return "1-10"  # a single location says nothing about size either way
    if n <= 10:
        return "11-50"
    if n <= 50:
        return "201-500"
    return "1000+"

# Direct range mentions: "51 to 200 employees", "50-200 staff", "2,500-5,000
# employees", "1,000-5,000 associates" (same "Associates" rationale as above)
_RANGE_PATTERN = re.compile(
    rf"\b{_NUM}\s*(?:to|-)\s*{_NUM}\s+(?:employees?|associates?|staff|team\s+members?|professionals?)\b",
    re.I,
)


def _midpoint(lo: int, hi: int) -> int:
    return (lo + hi) // 2


def _parse_int(raw: str) -> int | None:
    """Parse a captured number fragment (may contain comma thousands separators)."""
    try:
        return int(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def extract_employee_count(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Extract employee range from crawled pages.

    Confidence is the pattern's base confidence weighted by page type
    (About > Team > Careers > Homepage > other) — see _page_weight().
    The single highest weighted match across all pages and patterns wins,
    so a precise mention on a low-priority page can still lose to a
    less-specific mention on a high-priority page, and vice versa.

    Returns at most one ExtractedFact (highest weighted-confidence match).
    """
    best: tuple[str, float, str, str] | None = None  # (range, weighted_conf, url, source_type)

    for page in pages:
        if not page.ok:
            continue
        try:
            soup = BeautifulSoup(page.html, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
        except Exception:
            continue

        source_type = source_type_for_url(page.url)
        weight = _page_weight(source_type)

        # Range patterns first (more specific = higher base confidence)
        for match in _RANGE_PATTERN.finditer(text):
            lo = _parse_int(match.group(1))
            hi = _parse_int(match.group(2))
            if lo is None or hi is None or lo < 1 or lo > hi or hi > 500_000:
                continue
            range_str = _normalize_count(_midpoint(lo, hi))
            weighted = 0.82 * weight
            if best is None or weighted > best[1]:
                best = (range_str, weighted, page.url, source_type)

        # Single-number patterns
        for pattern, confidence in _PATTERNS:
            for match in pattern.finditer(text):
                n = _parse_int(match.group(1))
                if n is None or n < 1 or n > 500_000:
                    continue
                range_str = _normalize_count(n)
                weighted = confidence * weight
                if best is None or weighted > best[1]:
                    best = (range_str, weighted, page.url, source_type)

        # Publicly-traded / stock-ticker self-description — see
        # _PUBLIC_COMPANY_PATTERN comment for rationale. Maps straight to
        # "1000+" since a genuine ticker mention is essentially unambiguous.
        for match in _PUBLIC_COMPANY_PATTERN.finditer(text):
            weighted = 0.90 * weight
            if best is None or weighted > best[1]:
                best = ("1000+", weighted, page.url, source_type)

        # Office/location-count pattern — separate normalization (see
        # _normalize_from_location_count) since it's a weaker, indirect
        # signal, not a stated headcount.
        for match in _LOCATION_PATTERN.finditer(text):
            n = _parse_int(match.group(1))
            if n is None or n < 1 or n > 500_000:
                continue
            range_str = _normalize_from_location_count(n)
            weighted = 0.65 * weight
            if best is None or weighted > best[1]:
                best = (range_str, weighted, page.url, source_type)

    if best:
        range_str, weighted_confidence, source_url, source_type = best
        return [ExtractedFact(
            field_name=F_EMPLOYEE_RANGE,
            value=range_str,
            source_url=source_url,
            source_type=source_type,
            confidence=round(min(weighted_confidence, 1.0), 3),
        )]

    return []