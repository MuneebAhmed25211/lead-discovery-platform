"""
crawlers/extractors/social.py
──────────────────────────────
Extract social media profile URLs from crawled pages.

Approach: scan all <a href> links for known social domain patterns.
Footer links are the most reliable source — most company sites put
social links in the footer nav.

Returns one fact per platform found (F_SOCIAL_LINKEDIN, F_SOCIAL_TWITTER, etc.)
Only stores profile/page URLs — not feed/post/share links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    ExtractedFact,
    F_SOCIAL_FACEBOOK,
    F_SOCIAL_INSTAGRAM,
    F_SOCIAL_LINKEDIN,
    F_SOCIAL_TWITTER,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult


@dataclass(frozen=True)
class SocialPattern:
    field_name: str
    pattern: re.Pattern
    # Pattern to REJECT — share buttons, not profile pages
    reject: re.Pattern | None = None


_SOCIAL_PATTERNS: list[SocialPattern] = [
    SocialPattern(
        field_name=F_SOCIAL_LINKEDIN,
        pattern=re.compile(r"linkedin\.com/company/[a-zA-Z0-9\-_%.]+", re.I),
        reject=re.compile(r"linkedin\.com/(in|pub|jobs|feed|search|posts|pulse)", re.I),
    ),
    SocialPattern(
        field_name=F_SOCIAL_TWITTER,
        pattern=re.compile(r"(?:twitter|x)\.com/[a-zA-Z0-9_]{1,50}(?:/)?$", re.I),
        reject=re.compile(r"(?:twitter|x)\.com/(share|intent|hashtag|search)", re.I),
    ),
    SocialPattern(
        field_name=F_SOCIAL_FACEBOOK,
        pattern=re.compile(r"facebook\.com/(?:pages/)?[a-zA-Z0-9.\-_/]+", re.I),
        reject=re.compile(r"facebook\.com/(share|sharer|dialog|login|groups/discover)", re.I),
    ),
    SocialPattern(
        field_name=F_SOCIAL_INSTAGRAM,
        pattern=re.compile(r"instagram\.com/[a-zA-Z0-9._]{1,50}(?:/)?", re.I),
        reject=re.compile(r"instagram\.com/(p/|reel/|explore/|accounts/)", re.I),
    ),
]


def extract_social(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Extract social profile URLs from crawled pages.
    Returns one fact per social platform found.
    """
    found: dict[str, ExtractedFact] = {}  # field_name → fact

    for page in pages:
        if not page.ok:
            continue
        if len(found) == len(_SOCIAL_PATTERNS):
            break  # all platforms found

        source_type = source_type_for_url(page.url)

        try:
            soup = BeautifulSoup(page.html, "lxml")
        except Exception:
            continue

        for a_tag in soup.find_all("a", href=True):
            href = str(a_tag.get("href", ""))
            if not href:
                continue

            for sp in _SOCIAL_PATTERNS:
                if sp.field_name in found:
                    continue
                if sp.reject and sp.reject.search(href):
                    continue
                match = sp.pattern.search(href)
                if match:
                    url = href.split("?")[0].rstrip("/")
                    if not url.startswith("http"):
                        url = f"https://{url.lstrip('/')}"
                    found[sp.field_name] = ExtractedFact(
                        field_name=sp.field_name,
                        value=url,
                        source_url=page.url,
                        source_type=source_type,
                        confidence=0.90,
                    )

    return list(found.values())
