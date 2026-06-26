"""
utils/validator.py
──────────────────
Deterministic company candidate validation layer.

This is the fix for the core discovery quality bug: raw search results
were being returned directly as "companies" without verifying they are
actually businesses. This module implements every validation rule from
the audit specification.

Validation runs in four passes, each O(1) or O(n) on small fixed sets:

  Pass 1 — TLD rejection         (.gov, .edu)
  Pass 2 — Domain rejection       (directories, aggregators, socials)
  Pass 3 — URL path rejection     (/appointment, /directory, /news, etc.)
  Pass 4 — Title rejection        (Association, Foundation, University, etc.)

A candidate must pass ALL four passes to be kept.

LinkedIn special handling:
  - linkedin.com is NOT rejected at the domain level (unlike Phase 3's
    original behavior).
  - linkedin.com/company/{slug} URLs ARE kept and stored as enrichment data.
  - linkedin.com/search, /feed, /posts, /jobs, /articles ARE rejected
    at the path level.
  - linkedin.com should never appear as a discovered company's primary
    domain — it is enrichment only. See _extract_linkedin_url().
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


# ── Pass 1: TLD rejection ─────────────────────────────────────────────────────

_REJECTED_TLDS = frozenset({".gov", ".edu", ".mil"})


def _fails_tld_check(domain: str) -> bool:
    """Reject .gov, .edu, .mil TLDs — these are never commercial prospects."""
    for tld in _REJECTED_TLDS:
        if domain.endswith(tld):
            return True
    return False


# ── Pass 2: Domain rejection ──────────────────────────────────────────────────

# Known non-company domains — directories, aggregators, socials, review sites.
# LinkedIn is intentionally NOT here (see module docstring).
_REJECTED_DOMAINS = frozenset({
    # Search engines
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
    # Social media (discovery use — enrichment is separate)
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "tiktok.com", "pinterest.com", "snapchat.com",
    # Professional networks used as directories — not prospect sites
    "glassdoor.com", "indeed.com", "ziprecruiter.com", "monster.com",
    "careerbuilder.com", "simplyhired.com",
    # Review and directory sites
    "yelp.com", "yellowpages.com", "whitepages.com", "bbb.org",
    "trustpilot.com", "g2.com", "capterra.com", "getapp.com",
    # Healthcare directories and booking aggregators
    "zocdoc.com", "healthgrades.com", "vitals.com", "webmd.com",
    "ratemds.com", "doximity.com", "practo.com",
    # General aggregators and content platforms
    "wikipedia.org", "wikimedia.org", "reddit.com", "quora.com",
    "medium.com", "substack.com", "wordpress.com", "blogger.com",
    "github.com", "stackoverflow.com", "crunchbase.com",
    "amazon.com", "ebay.com", "etsy.com",
    # News organisations — not prospects
    "nytimes.com", "wsj.com", "bloomberg.com", "reuters.com",
    "apnews.com", "bbc.com", "cnn.com", "forbes.com",
})

# Domain keyword patterns — reject any domain containing these substrings
_REJECTED_DOMAIN_KEYWORDS = (
    "directory", "directories", "listings", "yellowpages",
    "findadoctor", "doctorlist", "healthportal",
)


def _fails_domain_check(domain: str) -> bool:
    """Reject known non-company domains and domains with aggregator keywords."""
    if domain in _REJECTED_DOMAINS:
        return True
    for kw in _REJECTED_DOMAIN_KEYWORDS:
        if kw in domain:
            return True
    return False


# ── Pass 3: URL path rejection ────────────────────────────────────────────────

# Paths that indicate this URL is a resource page, not a company homepage.
# Applied BEFORE domain extraction so we still have path context.
_REJECTED_PATH_SEGMENTS = frozenset({
    "blog", "article", "articles", "news", "press",
    "resource", "resources", "faq", "faqs",
    "appointment", "appointments", "book", "booking", "schedule",
    "directory", "directories", "listing", "listings",
    "search", "find", "locate", "locator",
    "forum", "forums", "community", "discussion",
    "event", "events", "conference",
    "donate", "donation", "volunteer",
    "policy", "policies", "legal", "privacy", "terms",
    "login", "signin", "signup", "register", "account",
    # LinkedIn non-company paths
    "feed", "posts", "jobs", "pulse", "learning",
    "in",  "patient-portal", "patient_portal", "patientportal",
    "for-patients", "for_patients",
    "job-openings", "job_openings", "openings",
    "wp-content"# linkedin.com/in/{person} — personal profiles, not companies
})


def _fails_path_check(url: str) -> bool:
    """
    Reject URLs whose path indicates they are not a company's homepage.

    Checks the FIRST path segment after the domain — so
    example.com/blog/dental-tips is rejected (first segment = 'blog')
    but example.com/dental-services is NOT rejected (no rejected segment).
    """
    try:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        path = parsed.path.strip("/").lower()
    except Exception:
        return False

    if not path:
        return False

    first_segment = path.split("/")[0]
    return first_segment in _REJECTED_PATH_SEGMENTS


# ── Pass 4: Title rejection ───────────────────────────────────────────────────

# Title keywords that indicate a non-commercial entity.
_REJECTED_TITLE_PATTERNS = re.compile(
    r"\b("
    r"association|associations|"
    r"foundation|foundations|"
    r"university|universities|"
    r"college|colleges|"
    r"department|"
    r"government|"
    r"directory|"
    r"resource\s+center|"
    r"board\s+of|"
    r"public\s+health|"
    r"health\s+department|"
    r"county\s+of|"
    r"city\s+of|"
    r"state\s+of|"
    r"national\s+institute|"
    r"centers?\s+for\s+(disease|medicare)|"
    r"wikipedia|"
    r"zocdoc|"
    r"healthgrades|"
    r"webmd|"
    r"yelp"
    r")\b",
    re.IGNORECASE,
)


def _fails_title_check(title: str) -> bool:
    """Reject result titles that indicate non-commercial entities."""
    return bool(_REJECTED_TITLE_PATTERNS.search(title))


# ── LinkedIn enrichment extraction ────────────────────────────────────────────

_LINKEDIN_COMPANY_RE = re.compile(
    r"linkedin\.com/company/([a-zA-Z0-9\-_%]+)/?",
    re.IGNORECASE,
)


def extract_linkedin_url(url: str) -> str | None:
    """
    Extract a valid LinkedIn company page URL from any URL string.

    Returns the canonical URL (linkedin.com/company/{slug}) if the
    URL is a valid company page, None otherwise.

    Valid:   https://linkedin.com/company/acme-dental  → kept
    Invalid: https://linkedin.com/in/john-doe          → None (personal)
    Invalid: https://linkedin.com/jobs/view/123        → None (job listing)
    Invalid: https://linkedin.com/search/results/...   → None (search page)
    """
    match = _LINKEDIN_COMPANY_RE.search(url)
    if match:
        slug = match.group(1).rstrip("/")
        return f"https://www.linkedin.com/company/{slug}"
    return None


# ── Main public interface ─────────────────────────────────────────────────────

class ValidationResult:
    """Outcome of validating a single search result."""

    __slots__ = ("passed", "rejection_reason", "linkedin_url")

    def __init__(
        self,
        passed: bool,
        rejection_reason: str = "",
        linkedin_url: str = "",
    ) -> None:
        self.passed = passed
        self.rejection_reason = rejection_reason
        self.linkedin_url = linkedin_url


def validate_candidate(url: str, title: str, domain: str) -> ValidationResult:
    """
    Run all four validation passes against a single search result.

    Args:
        url:    The full source URL from the search provider.
        title:  The search result page title.
        domain: Already-normalized root domain (from utils/dedup.extract_domain).

    Returns:
        ValidationResult with .passed = True if all checks pass.
        If .passed = False, .rejection_reason explains which check failed.

    LinkedIn special case:
        linkedin.com/company/{slug} URLs are kept as enrichment data
        (stored in .linkedin_url) and are NOT treated as company domains.
        The candidate's primary domain must come from elsewhere.
    """
    # LinkedIn company pages — extract for enrichment but don't validate
    # the linkedin.com domain as the company's own site.
    linkedin_url = extract_linkedin_url(url)
    if domain == "linkedin.com" or domain.endswith(".linkedin.com"):
        if linkedin_url:
            # Valid company LinkedIn page — useful as enrichment, but this
            # result should not set the company's primary domain as linkedin.com.
            # Return as passed=False so it doesn't become a discovered_company
            # row, but the caller can still extract the linkedin_url for use.
            return ValidationResult(
                passed=False,
                rejection_reason="linkedin_enrichment_only",
                linkedin_url=linkedin_url,
            )
        return ValidationResult(
            passed=False,
            rejection_reason="linkedin_non_company_page",
        )

    # Pass 1: TLD
    if _fails_tld_check(domain):
        return ValidationResult(passed=False, rejection_reason=f"rejected_tld:{domain.split('.')[-1]}")

    # Pass 2: Domain
    if _fails_domain_check(domain):
        return ValidationResult(passed=False, rejection_reason=f"rejected_domain:{domain}")

    # Pass 3: URL path
    if _fails_path_check(url):
        try:
            first_seg = urlparse(url).path.strip("/").split("/")[0]
        except Exception:
            first_seg = "unknown"
        return ValidationResult(passed=False, rejection_reason=f"rejected_path:/{first_seg}")

    # Pass 4: Title
    if _fails_title_check(title):
        return ValidationResult(passed=False, rejection_reason=f"rejected_title:{title[:40]}")

    return ValidationResult(passed=True, linkedin_url=linkedin_url or "")
