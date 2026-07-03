"""
crawlers/extractors/name.py
────────────────────────────
Extract the company's canonical name from crawled pages.

Priority order (highest confidence first):
  1. schema.org Organization/subclass "name" via JSON-LD (incl. @graph)  (0.95)
  2. Open Graph og:site_name meta tag                                     (0.90)
  3. <title> tag — first or last segment before separator                 (0.75)
  4. <h1> on homepage — only if it looks like a real name                 (0.70)

Only the highest-confidence source is returned. We do NOT return
multiple name facts — one canonical name per company is the goal.
The research service takes the highest-confidence fact.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    SOURCE_HOMEPAGE,
    ExtractedFact,
    F_NAME,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult

# Separators commonly used in <title> tags
_TITLE_SEPARATORS = re.compile(r"\s*[\|\-\u2014\u2013\u00b7:]\s*")

# Legal suffixes that can be stripped (only when appended to a longer name)
_LEGAL_SUFFIXES = re.compile(
    r",?\s+(inc\.?|llc\.?|ltd\.?|corp\.?|co\.?|plc\.?|gmbh|s\.a\.?|pty\.?|pllc\.?|lp\.?|llp\.?)$",
    re.IGNORECASE,
)

# Generic words/phrases that are never a company name on their own
_GENERIC_TITLES = frozenset({
    "home", "homepage", "welcome", "index", "main", "official site",
    "official website", "website", "site", "page", "untitled",
    "untitled document", "new page", "default", "coming soon",
    "under construction", "404", "not found", "error", "403",
    "access denied", "forbidden",
})

# Headings that look like taglines or CTAs, not company names
_H1_REJECT_PATTERNS = re.compile(
    r"^(welcome|discover|explore|learn|find|get|book|schedule|contact|"
    r"call us|email us|about us|our story|who we are|what we do|"
    r"serving|providing|dedicated|committed|trusted|leading|premier|"
    r"your |we |our |the best|number one|#1)",
    re.IGNORECASE,
)

# Organization-like schema.org types (includes subclasses of Organization)
_SCHEMA_ORG_TYPES = frozenset({
    # Core
    "Organization", "Corporation", "LocalBusiness", "ProfessionalService",
    # Healthcare
    "MedicalBusiness", "MedicalClinic", "MedicalOrganization",
    "Dentist", "Physician", "Hospital", "MedicalSpecialty",
    "Optician", "Pharmacy", "DiagnosticLab",
    # Legal / Finance
    "LegalService", "Attorney", "AccountingService", "FinancialService",
    "InsuranceAgency",
    # Food / Hospitality
    "Restaurant", "FoodEstablishment", "Bakery", "CafeOrCoffeeShop",
    "Hotel", "LodgingBusiness",
    # Retail / Services
    "Store", "ShoppingCenter", "AutoDealer", "AutoRepair",
    "BeautySalon", "HealthClub", "SportsActivityLocation",
    "HomeAndConstructionBusiness", "Plumber", "Electrician",
    "RealEstateAgent", "TravelAgency", "EducationalOrganization",
    "School", "CollegeOrUniversity",
    # Tech / Consulting
    "SoftwareApplication", "ITCompany", "ConsultingBusiness",
    "MarketingAgency", "RecruitmentAgency",
})


def _clean(name: str) -> str:
    """Normalise whitespace and strip outer quotes."""
    name = name.strip().strip('"\'')
    name = re.sub(r"\s+", " ", name)
    return name


def _is_valid_name(name: str) -> bool:
    """
    Return True if the string looks like a real company name.

    Rejects:
    - Too short or too long
    - Pure generics ("Home", "Welcome", etc.)
    - Strings that are only numbers/punctuation
    - Strings with obvious non-name patterns (URLs, emails)
    """
    if not name:
        return False
    if len(name) < 2 or len(name) > 100:
        return False
    if name.lower() in _GENERIC_TITLES:
        return False
    # Reject if it looks like a URL or email
    if re.search(r"https?://|www\.|@", name):
        return False
    # Must contain at least one letter
    if not re.search(r"[a-zA-Z]", name):
        return False
    return True


def _is_homepage(url: str) -> bool:
    """
    Deterministic homepage detection.

    A URL is a homepage if its path is empty, "/", or contains no
    meaningful path segments after stripping trailing slashes.

    Handles:
      https://example.com           → True
      https://example.com/          → True
      https://www.example.com/      → True
      https://example.com/about     → False
      https://example.com/about-us  → False
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        return path == "" or path == "/"
    except Exception:
        return False


def _from_schema_org(soup: BeautifulSoup) -> str | None:
    """
    Extract name from schema.org JSON-LD blocks.

    Handles:
      - Single object: {"@type": "Organization", "name": "..."}
      - Array of objects: [{"@type": "Organization", ...}, ...]
      - @graph wrapper: {"@graph": [{"@type": "Organization", ...}]}
      - Multiple @type values: {"@type": ["Organization", "LocalBusiness"]}
      - Organization subclasses: Dentist, Attorney, etc.
      - Alternate name fields: legalName, alternateName, brand.name
    """
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            raw = tag.string or ""
            if not raw.strip():
                continue
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        # Flatten: handle top-level list or @graph
        items: list = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "@graph" in data:
                graph = data["@graph"]
                items = graph if isinstance(graph, list) else [graph]
            else:
                items = [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            type_val = item.get("@type", "")
            if isinstance(type_val, str):
                types = [type_val]
            elif isinstance(type_val, list):
                types = type_val
            else:
                continue

            if not any(t in _SCHEMA_ORG_TYPES for t in types):
                continue

            # Try name fields in priority order: name > legalName >
            # brand.name > alternateName
            name_candidates = [item.get("name", "")]

            legal_name = item.get("legalName", "")
            if legal_name:
                name_candidates.append(legal_name)

            brand = item.get("brand", {})
            if isinstance(brand, dict):
                brand_name = brand.get("name", "")
                if brand_name:
                    name_candidates.append(brand_name)
            elif isinstance(brand, str):
                name_candidates.append(brand)

            alt_name = item.get("alternateName", "")
            if alt_name:
                name_candidates.append(alt_name)

            for candidate in name_candidates:
                if candidate and isinstance(candidate, str):
                    cleaned = _clean(candidate)
                    if _is_valid_name(cleaned):
                        return cleaned

    return None


def _from_twitter_meta(soup: BeautifulSoup) -> str | None:
    """
    Extract from Twitter Card metadata.

    twitter:title often mirrors og:title but is sometimes present when
    Open Graph tags are missing entirely (some site builders only emit one).
    twitter:site is occasionally a brand handle, useful only as a last
    resort signal, not used here as it's frequently just a generic handle.
    """
    tag = soup.find("meta", attrs={"name": "twitter:title"})
    if tag and tag.get("content"):
        content = str(tag["content"]).strip()
        parts = _TITLE_SEPARATORS.split(content)
        candidate = _clean(parts[0]) if parts else ""
        if _is_valid_name(candidate) and candidate.lower() not in _GENERIC_TITLES:
            return candidate
    return None


def _from_logo_alt_text(soup: BeautifulSoup) -> str | None:
    """
    Extract from the alt text of an image that looks like the site logo.

    Many sites without schema.org or clean OG tags still have
    <img alt="Acme Dental Group Logo" class="logo"> in the header,
    which is a reliable deterministic signal.
    """
    logo_imgs = soup.find_all(
        "img",
        attrs={"class": re.compile(r"logo", re.I)},
    ) + soup.find_all(
        "img",
        attrs={"id": re.compile(r"logo", re.I)},
    )

    for img in logo_imgs[:5]:
        alt = img.get("alt", "")
        if not alt:
            continue
        # Strip common trailing words like "Logo", "logo image"
        cleaned = re.sub(r"\s*\blogo\b.*$", "", alt, flags=re.IGNORECASE).strip()
        cleaned = _clean(cleaned)
        if cleaned and _is_valid_name(cleaned):
            return cleaned

    return None


def _from_apple_app_title(soup: BeautifulSoup) -> str | None:
    """
    Extract from apple-mobile-web-app-title meta tag.

    This is a real, commonly-present metadata field (used as the label
    when a site is added to an iOS home screen) that frequently contains
    a clean, short brand name distinct from the page <title>.
    """
    tag = soup.find("meta", attrs={"name": "apple-mobile-web-app-title"})
    if tag and tag.get("content"):
        cleaned = _clean(str(tag["content"]))
        if _is_valid_name(cleaned) and cleaned.lower() not in _GENERIC_TITLES:
            return cleaned
    return None
    """
    Extract from a prominent header/brand element's text content.

    Common site-builder patterns wrap the brand name in an element with
    a class like "site-title", "brand", "navbar-brand", or "logo-text".
    This is a lower-confidence fallback used only when schema.org, OG,
    title, and h1 all fail.
    """
    candidates = soup.find_all(
        attrs={"class": re.compile(r"site-title|brand-name|navbar-brand|logo-text|company-name", re.I)},
    )
    for el in candidates[:5]:
        text = _clean(el.get_text(separator=" ", strip=True))
        if text and len(text) <= 60 and _is_valid_name(text):
            return text
    return None


def _from_og_site_name(soup: BeautifulSoup) -> str | None:
    """Extract from og:site_name meta tag."""
    tag = soup.find("meta", property="og:site_name")
    if tag and tag.get("content"):
        cleaned = _clean(str(tag["content"]))
        if _is_valid_name(cleaned):
            return cleaned

    # Fallback: og:title — less authoritative but sometimes useful
    tag = soup.find("meta", property="og:title")
    if tag and tag.get("content"):
        content = str(tag["content"]).strip()
        # og:title often has "Company Name | Page Name" format
        parts = _TITLE_SEPARATORS.split(content)
        candidate = _clean(parts[0]) if parts else ""
        if _is_valid_name(candidate) and candidate.lower() not in _GENERIC_TITLES:
            return candidate

    return None


def _from_title(soup: BeautifulSoup) -> str | None:
    """
    Extract company name from the <title> tag.

    Strategy:
      1. Split on common separators (|, -, —, ·, :)
      2. Try first segment — most sites put company name first
      3. If first segment is generic, try last segment
      4. Validate before returning
    """
    title_tag = soup.find("title")
    if not title_tag:
        return None
    raw = (title_tag.string or "").strip()
    if not raw:
        return None

    parts = [p.strip() for p in _TITLE_SEPARATORS.split(raw) if p.strip()]
    if not parts:
        return None

    # Try first segment
    candidate = _clean(parts[0])
    if _is_valid_name(candidate) and candidate.lower() not in _GENERIC_TITLES:
        return candidate

    # Try last segment (some sites put name last: "Home | Acme Corp")
    if len(parts) > 1:
        candidate = _clean(parts[-1])
        if _is_valid_name(candidate) and candidate.lower() not in _GENERIC_TITLES:
            return candidate

    return None


def _from_h1(soup: BeautifulSoup) -> str | None:
    """
    Extract company name from the first <h1> on the page.

    Only used as a last resort. Applies strict validation:
    - Must not match CTA/tagline patterns
    - Must be short enough to be a name (≤ 60 chars)
    - Must not contain sentence-level punctuation (. ! ?)
      that indicates it is a sentence, not a name
    """
    for h1 in soup.find_all("h1"):
        text = _clean(h1.get_text(separator=" ", strip=True))
        if not text:
            continue
        if len(text) > 60:
            continue
        # Reject if it looks like a tagline or CTA
        if _H1_REJECT_PATTERNS.match(text):
            continue
        # Reject if it contains sentence punctuation
        if re.search(r"[.!?]", text):
            continue
        if _is_valid_name(text):
            return text

    return None


def extract_name(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Extract company name facts from crawled pages.

    Returns a list with at most one ExtractedFact (the highest-confidence
    name found). Homepage is tried first, then about page.

    Tries all four sources in priority order for each page before moving
    to the next page — so schema.org on the about page beats <h1> on
    the homepage.
    """
    # Homepage pages first, then about pages, then others
    def page_priority(p: PageResult) -> int:
        if _is_homepage(p.url):
            return 0
        src = source_type_for_url(p.url)
        if src == "company_about":
            return 1
        return 2

    sorted_pages = sorted(
        (p for p in pages if p.ok),
        key=page_priority,
    )

    best_value: str | None = None
    best_confidence: float = 0.0
    best_url: str = ""
    best_source_type: str = SOURCE_HOMEPAGE

    for page in sorted_pages:
        try:
            soup = BeautifulSoup(page.html, "lxml")
        except Exception:
            continue

        source_type = source_type_for_url(page.url)

        # Only extract name from homepage or about page
        if page_priority(page) > 1:
            continue

        candidates: list[tuple[str | None, float]] = [
            (_from_schema_org(soup), 0.95),
            (_from_og_site_name(soup), 0.90),
            (_from_apple_app_title(soup), 0.85),
            (_from_twitter_meta(soup), 0.80),
            (_from_title(soup), 0.75),
            (_from_h1(soup), 0.70),
            (_from_logo_alt_text(soup), 0.68),
            (_from_header_brand_text(soup), 0.60),
        ]

        for value, confidence in candidates:
            if value and confidence > best_confidence:
                best_value = value
                best_confidence = confidence
                best_url = page.url
                best_source_type = source_type
                # schema.org or og:site_name on any page — stop immediately
                if confidence >= 0.90:
                    break

        # If we already have a high-confidence result, stop paging
        if best_confidence >= 0.90:
            break

    if best_value:
        normalized = _strip_trailing_legal_suffix(best_value)
        return [ExtractedFact(
            field_name=F_NAME,
            value=normalized,
            source_url=best_url,
            source_type=best_source_type,
            confidence=best_confidence,
        )]

    return []


def _strip_trailing_legal_suffix(name: str) -> str:
    """
    Strip a trailing legal suffix (Inc, LLC, Ltd, Corp, etc.) ONLY when
    it is appended after a comma or clearly separable, and only if doing
    so leaves a name of reasonable length.

    "Acme Dental Group, LLC" -> "Acme Dental Group"
    "Acme LLC"               -> "Acme" (kept short names as-is if too short)
    "Smile Care Dental"      -> unchanged (no suffix present)

    This is purely cosmetic normalization — it does not affect which
    candidate was selected, only the final stored value.
    """
    stripped = _LEGAL_SUFFIXES.sub("", name).strip().rstrip(",").strip()
    # Don't strip if it would leave an unreasonably short name
    if len(stripped) >= 2:
        return stripped
    return name
