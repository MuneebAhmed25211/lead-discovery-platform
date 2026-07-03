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

# US state abbreviations for pattern matching
_US_STATES = re.compile(
    r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|"
    r"MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|"
    r"UT|VT|VA|WA|WV|WI|WY|DC)\b"
)

# Patterns that look like "City, ST" or "City, State" or "City, Country"
_CITY_STATE_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z\s]{2,25}),\s*([A-Z]{2}|[A-Z][a-z]{3,20})\b"
)

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
        # Look for city/state patterns
        for match in _CITY_STATE_PATTERN.finditer(text):
            city = match.group(1).strip()
            state = match.group(2).strip()
            # Filter out false positives (very generic words)
            if city.lower() in {"we", "our", "the", "and", "for", "with"}:
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
