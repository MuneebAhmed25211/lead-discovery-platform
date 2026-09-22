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

import html
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
_TITLE_SEPARATORS = re.compile(r"\s+-\s+|\s*[\|\u2014\u2013\u00b7:]\s*")

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
    # Single generic words that surface from truncated nav links, filter
    # labels, or category headers ("All Services", "All Products") rather
    # than an actual company name. Confirmed real case: "All" extracted
    # for smartmoving.com.
    "all", "shop", "services", "solutions", "products", "menu", "about",
    "contact", "blog", "news", "careers", "team",
})

# Headings that look like taglines or CTAs, not company names
_H1_REJECT_PATTERNS = re.compile(
    r"^(welcome|discover|explore|learn|find|get|book|schedule|contact|"
    r"call us|email us|about us|our story|who we are|what we do|"
    r"serving|providing|dedicated|committed|trusted|leading|premier|"
    r"your |we |our |the best|number one|#\s*1|a\s+top\s+\d|the\s+top\s+\d|"
    # Added: confirmed real case, myboutiquehotel.com's og:site_name was
    # "The N.1 to find and book the best boutique hotels around the
    # world" — a ranking-claim opener ("The N.1"/"The No.1") followed by
    # a verb phrase, which the existing alternatives didn't cover (they
    # only caught "the best"/"the top N" as openers, not "the N.1 to...").
    r"the\s+n\.?\s*1\s+to\b|the\s+no\.?\s*1\s+to\b"
    r")",
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
    """Normalise whitespace, decode HTML entities, and strip outer quotes."""
    name = html.unescape(name)
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
    - Tagline/accolade-shaped text ("A Top 15 'Global Elite' Firm",
      "#1 rated provider") — real company names don't read as headlines
      or contain embedded quotation marks. Previously this check
      (_H1_REJECT_PATTERNS) only ran on the h1-extraction path; the
      logo-alt-text and header-brand-text fallbacks had no equivalent
      guard, so a press/awards-badge image's alt text could pass
      through whole. Confirmed real case: King & Spalding.
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
    # Reject tagline/accolade-shaped text
    if _H1_REJECT_PATTERNS.match(name):
        return False
    # Reject embedded quotation marks (straight or curly)
    if re.search(r"[\"\u201c\u201d]", name):
        return False
    # Reject unrendered template placeholder text — confirmed real case:
    # a small-business site running an unconfigured template returned
    # the literal string "[Site Name]" as its <title>, which then
    # extracted verbatim as the company name and rendered as a visibly
    # broken result in the UI. Real company names don't come wrapped in
    # brackets, nor do they read as generic template tokens.
    if re.fullmatch(r"[\[\{<].*[\]\}>]", name.strip()):
        return False
    if re.search(
        r"\b(site\s*name|company\s*name|your\s+business\s+name|"
        r"business\s*name\s*here|untitled(?:\s+document)?|lorem\s+ipsum|"
        r"placeholder|insert\s+.*\s+here)\b",
        name,
        re.IGNORECASE,
    ):
        return False
    # WordPress's own default, un-customized title/tagline — same failure
    # shape as the "[Site Name]" placeholder check above (a template
    # default that was never replaced with the site owner's real name),
    # just specific to WordPress installs. Confirmed real case: "My
    # WordPress" was extracted and displayed as a company's name for
    # scaleupvoyager.com — a domain that shares no words with "WordPress"
    # at all, which is itself the tell that this was never a real brand
    # name. "Just another WordPress site" is WordPress's literal,
    # unmodified default tagline shown until a site owner changes it in
    # Settings -> General.
    if re.search(
        r"^my\s+wordpress(?:\s+(?:site|blog|website))?$|"
        r"just\s+another\s+wordpress\s+site",
        name,
        re.IGNORECASE,
    ):
        return False
    # Geo-blocking / access-denied / error-page boilerplate — a CDN or WAF
    # (Cloudflare, Akamai, and similar) returns a standardized block page
    # for a request it won't serve, and that block page's own <title> gets
    # extracted as if it were the company's name. Confirmed real case: a
    # dental/medical practice's site returned "Page is not available to
    # your current location." verbatim as its extracted, displayed name —
    # visibly broken output, not a wrong-but-plausible company name, the
    # same failure class as the WordPress/template-placeholder checks
    # above but for infrastructure-level block pages instead of
    # unconfigured CMS installs.
    if re.search(
        r"page\s+is\s+not\s+available|"
        r"(?:content|page)\s+(?:is\s+)?not\s+available\s+in\s+your|"
        r"not\s+available\s+(?:to|in)\s+your\s+(?:current\s+)?(?:location|region|country)|"
        r"access\s+denied|"
        r"you\s+don'?t\s+have\s+permission\s+to\s+access|"
        r"^(?:403|404)\s+(?:forbidden|not\s+found)$|"
        r"this\s+(?:content|page|site)\s+is\s+(?:currently\s+)?(?:unavailable|blocked)",
        name,
        re.IGNORECASE,
    ):
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
        if not cleaned:
            continue
        # Split on the same separators _from_title uses. Alt text on a
        # logo/badge image isn't guaranteed to BE just the name — e.g. an
        # awards-badge image's alt text can be a press mention like
        # 'A Top 15 "Global Elite" Firm - The American Lawyer', which
        # passed _is_valid_name whole because that check doesn't look for
        # embedded separators. Confirmed real case: King & Spalding.
        parts = [p.strip() for p in _TITLE_SEPARATORS.split(cleaned) if p.strip()]
        candidate = _clean(parts[0]) if parts else cleaned
        if _is_valid_name(candidate) and candidate.lower() not in _GENERIC_TITLES:
            return candidate

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


def _from_header_brand_text(soup: BeautifulSoup) -> str | None:
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
        if not text or len(text) > 60:
            continue
        # Same fix as _from_logo_alt_text: split on separators before
        # validating, don't trust that a "brand"-classed element's text
        # is already reduced to just the name.
        parts = [p.strip() for p in _TITLE_SEPARATORS.split(text) if p.strip()]
        candidate = _clean(parts[0]) if parts else text
        if _is_valid_name(candidate):
            return candidate
    return None


def _from_og_site_name(soup: BeautifulSoup) -> str | None:
    """Extract from og:site_name meta tag."""
    tag = soup.find("meta", property="og:site_name")
    if tag and tag.get("content"):
        raw = str(tag["content"])
        # Split on separators before validating — same fix already
        # applied to _from_logo_alt_text and _from_header_brand_text,
        # for the identical reason: this tag isn't guaranteed to already
        # be just the brand name. Confirmed real case: fandptravel.com
        # set og:site_name to "F&P Travel - The luxury travel experts"
        # (brand + tagline combined), which passed _is_valid_name whole
        # since it doesn't match any single-string rejection pattern —
        # only splitting it into segments and validating each one finds
        # the real name in the first segment.
        parts = [p.strip() for p in _TITLE_SEPARATORS.split(raw) if p.strip()]
        for part in parts:
            cleaned = _clean(part)
            if _is_valid_name(cleaned) and cleaned.lower() not in _GENERIC_TITLES:
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


def _domain_sld(domain: str) -> str:
    """Second-level domain name only, e.g. 'xerox' from 'xerox.com' or
    'www.xerox.co.uk'. Crude but effective — just needs to be good enough
    for substring token-matching, not a full public-suffix-list lookup."""
    parts = [p for p in domain.lower().split(".") if p and p != "www"]
    return parts[0] if parts else ""


def _matches_domain(candidate: str, sld: str) -> bool:
    """Shared by _from_title's own domain-preference pass and by
    extract_name's cross-source override below."""
    if not sld or len(sld) <= 2:
        return False
    tokens = re.findall(r"[a-z0-9]+", candidate.lower())
    return any(len(tok) > 2 and (sld in tok or tok in sld) for tok in tokens)


def _from_title(soup: BeautifulSoup, domain: str = "") -> str | None:
    """
    Extract company name from the <title> tag.

    Strategy:
      1. Split on common separators (|, -, —, ·, :)
      2. Validate every segment, not just first/last
      3. Among valid segments, prefer whichever shares a word with the
         domain — the real brand name is almost always reflected in the
         domain (xerox.com -> "Xerox"), while a marketing tagline
         ("Innovation at Work", "Apartments & Homes for Rent") typically
         isn't. Confirmed real cases this resolves: Xerox extracted as
         "Innovation at Work", Greystar extracted as "Apartments & Homes
         for Rent" — the correct name was present as another title
         segment both times, but the wrong one already passed validation
         and was returned first, since taglines like these don't trigger
         any of _is_valid_name's rejection patterns.
      4. No domain match found (or no domain given) — fall back to the
         first valid segment, same as the original behavior.
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

    valid_candidates = []
    for part in parts:
        candidate = _clean(part)
        if _is_valid_name(candidate) and candidate.lower() not in _GENERIC_TITLES:
            valid_candidates.append(candidate)

    if not valid_candidates:
        return None

    sld = _domain_sld(domain) if domain else ""
    if sld and len(sld) > 2:
        for candidate in valid_candidates:
            if _matches_domain(candidate, sld):
                return candidate

    # No domain match — fall back to the original first-then-last bias
    # (company names are overwhelmingly at the start or end of a title,
    # rarely buried in the middle), rather than just the first *valid*
    # segment found in scan order, which can accidentally prefer a middle
    # segment over the real last-segment company name.
    first_candidate = _clean(parts[0])
    if _is_valid_name(first_candidate) and first_candidate.lower() not in _GENERIC_TITLES:
        return first_candidate
    if len(parts) > 1:
        last_candidate = _clean(parts[-1])
        if _is_valid_name(last_candidate) and last_candidate.lower() not in _GENERIC_TITLES:
            return last_candidate

    return valid_candidates[0]


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
    best_soup: BeautifulSoup | None = None
    best_page_domain: str = ""

    for page in sorted_pages:
        try:
            soup = BeautifulSoup(page.html, "lxml")
        except Exception:
            continue

        source_type = source_type_for_url(page.url)
        page_domain = urlparse(page.url).netloc

        # Only extract name from homepage or about page
        if page_priority(page) > 1:
            continue

        candidates: list[tuple[str | None, float]] = [
            (_from_schema_org(soup), 0.95),
            (_from_og_site_name(soup), 0.90),
            (_from_apple_app_title(soup), 0.85),
            (_from_twitter_meta(soup), 0.80),
            (_from_title(soup, domain=page_domain), 0.75),
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
                best_soup = soup
                best_page_domain = page_domain
                # schema.org or og:site_name on any page — stop immediately
                if confidence >= 0.90:
                    break

        # If we already have a high-confidence result, stop paging
        if best_confidence >= 0.90:
            break

    # Cross-check the winning candidate against the domain, regardless of
    # which source produced it. _from_title already prefers a
    # domain-matching title segment over a mismatched one when it's
    # comparing segments of the SAME title — but schema.org/og:site_name/
    # apple-app-title/twitter-meta (all higher-priority sources, tried
    # first above) never get that same cross-check, since they win and
    # the loop exits before _from_title is even reached. Confirmed real
    # case: adm.com's og:site_name for its homepage is "Human, Pet and
    # Animal Nutrition Company" (a business-unit page title, not the
    # corporate brand), but that same page's own <title> tag is "Human,
    # Pet and Animal Nutrition Company | ADM" — "ADM" is right there,
    # sharing an exact token with the domain, and would have won under
    # _from_title's own preference logic had it been reached. This
    # override runs that same preference check one level up, across
    # sources rather than only within <title>'s own segments.
    if best_value and best_soup is not None and not _matches_domain(best_value, _domain_sld(best_page_domain)):
        title_override = _from_title(best_soup, domain=best_page_domain)
        if title_override and title_override != best_value and _matches_domain(
            title_override, _domain_sld(best_page_domain)
        ):
            best_value = title_override
            best_confidence = max(best_confidence, 0.75)

    if best_value:
        normalized = _strip_trailing_page_descriptor(_strip_trailing_legal_suffix(best_value))
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


_TRAILING_PAGE_DESCRIPTORS = re.compile(
    r",?\s+(homepage|home\s*page|official\s+site|official\s+website|"
    r"main\s+page)$",
    re.IGNORECASE,
)


def _strip_trailing_page_descriptor(name: str) -> str:
    """
    Strip a trailing generic page-descriptor word ("Homepage", "Official
    Site", etc.) the same way _strip_trailing_legal_suffix (above) strips
    "Inc"/"LLC" — these describe the PAGE, never the BRAND, so no real
    company's actual name ends with one.

    Confirmed real case: "Bertelsmann Homepage" was extracted and
    displayed as a company's name. Same failure shape as the WordPress-
    default-placeholder fix elsewhere in this file (a template/generic
    word stuck onto an otherwise real name) — just a SUFFIX rather than a
    whole-string match, so the existing exact-match _GENERIC_TITLES
    rejection (which already lists "homepage" alone) never catches it:
    "Bertelsmann Homepage" != "homepage".

    "Bertelsmann Homepage"    -> "Bertelsmann"
    "Acme Corp Official Site" -> "Acme Corp"
    "Bertelsmann"             -> unchanged (no descriptor present)
    """
    stripped = _TRAILING_PAGE_DESCRIPTORS.sub("", name).strip().rstrip(",").strip()
    if len(stripped) >= 2:
        return stripped
    return name