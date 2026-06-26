"""
utils/dedup.py
──────────────
Deterministic deduplication utilities for company lists.

Deduplication is 100% rule-based — no LLM involved.

Strategies applied in order:
  1. Exact domain match          — "acme.com" == "acme.com"
  2. Root domain match           — "blog.acme.com" == "app.acme.com"
  3. Name normalisation match    — "Acme Corp" == "ACME CORPORATION"

The first occurrence of a domain is kept; subsequent duplicates are dropped.
A deduplicated list preserves the original ordering (first-seen wins).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


def extract_domain(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
    except Exception:
        hostname = url.lower()
    hostname = re.sub(r"^(www\d?|blog|app|help|support|docs|careers|jobs)\.", "", hostname)
    hostname = hostname.split(":")[0]
    return hostname.strip("/")


def root_domain(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) >= 2:
        if parts[-2] in ("co", "com", "org", "net", "gov") and len(parts[-1]) == 2:
            return ".".join(parts[-3:]) if len(parts) >= 3 else domain
        return ".".join(parts[-2:])
    return domain


_COMPANY_SUFFIXES = re.compile(
    r"\b(inc\.?|llc\.?|ltd\.?|corp\.?|corporation|limited|group|"
    r"holdings?|international|global|solutions?|technologies?|services?|"
    r"consulting|digital|agency|studio|labs?|co\.?)\b",
    re.IGNORECASE,
)


def normalise_company_name(name: str) -> str:
    name = name.lower()
    name = _COMPANY_SUFFIXES.sub("", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


_BLACKLISTED_DOMAINS = frozenset({
    "linkedin.com", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "youtube.com", "wikipedia.org", "crunchbase.com",
    "glassdoor.com", "indeed.com", "reddit.com", "quora.com",
    "medium.com", "github.com", "stackoverflow.com", "amazon.com",
    "google.com", "bing.com", "yahoo.com", "yelp.com", "trustpilot.com",
})


def _is_blacklisted_domain(domain: str) -> bool:
    return root_domain(domain) in _BLACKLISTED_DOMAINS


def deduplicate_raw_companies(companies: list[dict]) -> list[dict]:
    """
    Remove duplicate companies from a discovery result list.

    Each company dict must have at least:
      - "domain": str
      - "name": str

    Returns a new list with duplicates removed, preserving order (first-seen wins).
    """
    seen_roots: set[str] = set()
    seen_names: set[str] = set()
    result: list[dict] = []

    for company in companies:
        raw_domain = company.get("domain", "")
        if not raw_domain:
            raw_domain = extract_domain(company.get("website", ""))

        domain = extract_domain(raw_domain)
        root = root_domain(domain)
        norm_name = normalise_company_name(company.get("name", ""))

        if root and root in seen_roots:
            continue
        if norm_name and len(norm_name) > 3 and norm_name in seen_names:
            continue
        if _is_blacklisted_domain(domain):
            continue

        seen_roots.add(root)
        if norm_name:
            seen_names.add(norm_name)

        company = dict(company)
        company["domain"] = domain
        result.append(company)

    return result
