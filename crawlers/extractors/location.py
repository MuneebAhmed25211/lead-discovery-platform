"""
crawlers/extractors/location.py
────────────────────────────────
Extract location (city, state, country) from crawled pages.

Priority order:
  1. schema.org address.addressLocality + addressRegion  (0.95)
  2. Open Graph og:locale mapped to country              (0.60)
  3. Footer address patterns — regex on visible text     (0.70)
  4. Contact page address patterns                       (0.75)

Returns one fact per distinct location found (a company can have
multiple locations — each becomes a separate evidence row).
"""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    ExtractedFact,
    F_LOCATION,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult

# US state abbreviations for pattern validation
_US_STATE_CODES = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
})

# Full country/region names that legitimately follow "City, X" in an
# address (as opposed to a US state code). Deliberately not exhaustive —
# scoped to countries this platform's ICPs actually target (see
# core/config.py rate limiting / discovery target countries), extended
# only as real addresses surface a country not yet in this list.
_COUNTRY_OR_REGION_NAMES = frozenset({
    "usa", "united states", "us", "uk", "united kingdom", "england",
    "scotland", "wales", "canada", "australia", "ireland", "new zealand",
})

# A state/country token: either a 2-letter US state code, or a single
# capitalized word up to 20 letters (covers "Australia", "Canada", etc.
# from _COUNTRY_OR_REGION_NAMES — validated, not just pattern-matched).
_STATE_TOKEN_PATTERN = re.compile(r"^([A-Za-z]{2,20})\b")

# A city-like trailing word/phrase at the end of a text segment.
_CITY_TOKEN_PATTERN = re.compile(r"([A-Z][a-zA-Z]{2,25})$")

# Zip code adjacent to city/state — helps confirm it's a real address
_ZIP_PATTERN = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def _from_schema_org(soup: BeautifulSoup, source_url: str) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    source_type = source_type_for_url(source_url)

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            address = item.get("address") or item.get("location", {})
            if isinstance(address, str):
                facts.append(ExtractedFact(
                    field_name=F_LOCATION,
                    value=address.strip(),
                    source_url=source_url,
                    source_type=source_type,
                    confidence=0.85,
                ))
                continue
            if not isinstance(address, dict):
                continue

            parts = []
            locality = address.get("addressLocality", "")
            region = address.get("addressRegion", "")
            country = address.get("addressCountry", "")

            if locality:
                parts.append(locality)
            if region:
                parts.append(region)
            if country and country not in parts:
                # Normalize ISO country codes
                if country == "US":
                    country = "United States"
                elif country == "GB":
                    country = "United Kingdom"
                elif country == "AU":
                    country = "Australia"
                elif country == "CA":
                    country = "Canada"
                elif country == "DE":
                    country = "Germany"
                parts.append(country)

            if parts:
                facts.append(ExtractedFact(
                    field_name=F_LOCATION,
                    value=", ".join(parts),
                    source_url=source_url,
                    source_type=source_type,
                    confidence=0.95,
                ))

    return facts


def _from_footer_text(soup: BeautifulSoup, source_url: str) -> list[ExtractedFact]:
    """
    Look for address patterns in footer, contact sections, and address tags.

    Scans every adjacent pair of comma-separated segments in the element's
    text (rather than one regex spanning the whole string) because most
    real addresses have a street segment before the city — e.g.
    "123 Main St, Austin, TX 78701" — and a single greedy capture group
    mismatches on exactly that shape, swallowing "Main St, Austin" as one
    segment and never finding "Austin, TX" at all. Checking each pair lets
    the correct (city, state) segment win regardless of how many commas
    precede it, while still requiring the second segment of the pair to
    validate as a real state code or country name (rejecting shapes like
    a byline — "Lee Ying Shan, Sarah" — or a name/department pair —
    "Ehsan, Engineering" — both confirmed false positives from real
    crawls of CNBC and Apple).
    """
    facts: list[ExtractedFact] = []
    source_type = source_type_for_url(source_url)

    # Check <address> tags and footer elements
    address_elements = (
        soup.find_all("address") +
        soup.find_all(attrs={"class": re.compile(r"footer|address|location|contact", re.I)})
    )

    seen: set[str] = set()
    for el in address_elements[:5]:  # cap to avoid huge DOMs
        text = el.get_text(" ", strip=True)
        segments = [s.strip() for s in text.split(",") if s.strip()]

        for i in range(len(segments) - 1):
            state_match = _STATE_TOKEN_PATTERN.match(segments[i + 1])
            if not state_match:
                continue
            state = state_match.group(1)
            # Exact-case match for 2-letter codes, not case-insensitive:
            # several state abbreviations (IN, OR, ME, HI, OK, CO, PA...)
            # double as common English words/abbreviations that appear
            # lowercase in ordinary prose ("...Solutions, in the
            # industry..."), while real addresses are virtually always
            # written in full caps ("Austin, TX"). Country names stay
            # case-insensitive since they appear in prose in any casing.
            # Confirmed real cases this fixes: "CVS, in" and "Solutions,
            # in" both matched Indiana via case-insensitive "in" == "IN".
            is_state_code = len(state) == 2 and state in _US_STATE_CODES
            if not is_state_code and state.lower() not in _COUNTRY_OR_REGION_NAMES:
                continue

            city_match = _CITY_TOKEN_PATTERN.search(segments[i])
            if not city_match:
                continue
            city = city_match.group(1)
            # Filter out false positives (very generic words, street
            # suffixes, and job titles). Job titles are the confirmed
            # real case: EduGrowth's crawled page produced "CEO,
            # Australia" as a location — a leadership-bio byline (e.g.
            # "Jane Smith, CEO, Australia") where "CEO" happened to sit
            # immediately before a real country name, the exact same
            # byline/title-adjacent-to-real-location shape already
            # documented above for "Lee Ying Shan, Sarah" and "Ehsan,
            # Engineering" — just a job title this set didn't cover yet.
            if city.lower() in {
                "we", "our", "the", "and", "for", "with", "st", "ave",
                "rd", "dr", "blvd", "ceo", "cto", "cfo", "coo", "cmo",
                "cio", "founder", "cofounder", "president", "chairman",
                "chairwoman", "chair", "director", "manager", "vp",
            }:
                continue

            location = f"{city}, {state}"
            if location not in seen:
                seen.add(location)
                # Higher confidence if a zip code is nearby
                confidence = 0.80 if _ZIP_PATTERN.search(text) else 0.65
                facts.append(ExtractedFact(
                    field_name=F_LOCATION,
                    value=location,
                    source_url=source_url,
                    source_type=source_type,
                    confidence=confidence,
                ))

    return facts


def extract_location(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Extract location facts from crawled pages.
    Returns one fact per distinct location found. De-duplicated by value.
    """
    all_facts: list[ExtractedFact] = []
    seen_values: set[str] = set()

    for page in pages:
        if not page.ok:
            continue
        try:
            soup = BeautifulSoup(page.html, "lxml")
        except Exception:
            continue

        page_facts = (
            _from_schema_org(soup, page.url) +
            _from_footer_text(soup, page.url)
        )

        for fact in page_facts:
            if fact.value and fact.value not in seen_values:
                seen_values.add(fact.value)
                all_facts.append(fact)

    # Return highest-confidence facts first
    return sorted(all_facts, key=lambda f: f.confidence, reverse=True)