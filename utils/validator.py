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
    # Job boards / staffing & recruiting agencies — these are themselves
    # businesses, but they are noise relative to whatever industry an ICP
    # query targets (a "law firms expanding" query returning a staffing
    # agency is off-target, not a lead). Named explicitly per audit
    # findings (e.g. "Robert Half Jobs", "Adzuna").
    "roberthalf.com", "adecco.com", "randstad.com", "manpowergroup.com",
    "kellyservices.com", "michaelpage.com", "hays.com", "adzuna.com",
    "dice.com", "themuse.com", "lever.co", "greenhouse.io",
    "myworkdayjobs.com", "icims.com", "jobvite.com", "smartrecruiters.com",
    "totaljobs.com", "reed.co.uk", "cv-library.co.uk", "careerjet.com",
    "jobstreet.com", "jobsdb.com", "seek.com.au", "seek.co.nz",
    "jooble.org", "talent.com", "snagajob.com", "flexjobs.com",
    # Confirmed in production discovery output (see audit evidence):
    # staffing agencies and a niche industry job board.
    "insightglobal.com", "appleone.com", "spacecrew.com",
    # Review and directory sites
    "yelp.com", "yellowpages.com", "whitepages.com", "bbb.org",
    "trustpilot.com", "g2.com", "capterra.com", "getapp.com",
    # Legal directories / publishers — not the law firms themselves
    "findlaw.com", "avvo.com", "justia.com", "martindale.com",
    "lawyers.com", "superlawyers.com", "nolo.com",
    # Healthcare directories and booking aggregators
    "zocdoc.com", "healthgrades.com", "vitals.com", "webmd.com",
    "ratemds.com", "doximity.com", "practo.com",
    # General aggregators and content platforms
    "wikipedia.org", "wikimedia.org", "reddit.com", "quora.com",
    "medium.com", "substack.com", "wordpress.com", "blogger.com",
    "github.com", "stackoverflow.com", "crunchbase.com",
    "amazon.com", "ebay.com", "etsy.com",
    # Relocation / real-estate content sites named in audit findings
    "migratemate.co", "migratemate.com",
    # Trade-showcase / portfolio content sites named in audit findings
    "officesnapshots.com",
    # News organisations and press-release wire services — not prospects
    "nytimes.com", "wsj.com", "bloomberg.com", "reuters.com",
    "apnews.com", "bbc.com", "cnn.com", "forbes.com",
    "thebusinessdesk.com", "prnewswire.com", "businesswire.com",
    "globenewswire.com", "techcrunch.com", "businessinsider.com",
    "entrepreneur.com", "inc.com", "fastcompany.com",
    # Confirmed in production discovery output: a legal-industry news
    # publisher and a careers/employer-rankings directory site.
    "law.com", "vault.com",
})

# Domain keyword patterns — reject any domain containing these substrings
_REJECTED_DOMAIN_KEYWORDS = (
    "directory", "directories", "listings", "yellowpages",
    "findadoctor", "doctorlist", "healthportal",
    # Job boards / staffing sites not already caught by exact-match above —
    # substring match catches regional/localized variants (e.g. "seek-jobs.co",
    # "xyzstaffing.com") that an exact-domain list can never fully enumerate.
    "jobs", "careers", "recruit", "staffing", "vacanc", "hiring",
)


def _fails_domain_check(domain: str) -> bool:
    """Reject known non-company domains and domains with aggregator keywords."""
    if domain in _REJECTED_DOMAINS:
        return True
    for kw in _REJECTED_DOMAIN_KEYWORDS:
        if kw in domain:
            return True
    return False


# ── Search-time noise reduction ───────────────────────────────────────────────
#
# A compact, curated subset of the highest-*volume* noise sources observed
# in production — large, generic job boards and press-release wires that
# show up across almost every ICP query, regardless of industry. Used to
# build a "-site:" exclusion suffix appended to search queries BEFORE they
# reach a provider — see services/discovery_service.py.
#
# Deliberately short: Serper proxies Google search, which has practical
# limits on query complexity, so a long exclusion suffix risks degrading
# result quality more than it helps. Narrower, industry-specific offenders
# named in audit findings (Robert Half, Adzuna, TheBusinessDesk, FindLaw,
# OfficeSnapshots, MigrateMate, etc.) are NOT duplicated here — they are
# already covered by the comprehensive _REJECTED_DOMAINS safety net at
# validation time (Pass 2), which has no query-length constraint. A
# candidate that slips past the search-time exclusion (e.g. from a
# provider that doesn't honor "-site:") is still rejected at validation.
_QUERY_EXCLUSION_DOMAINS = (
    "indeed.com", "glassdoor.com", "ziprecruiter.com", "monster.com",
    "linkedin.com/jobs", "prnewswire.com", "businesswire.com",
)


def negative_site_filters() -> str:
    """
    Build a Google-style '-site:domain' exclusion string from the curated
    noise-domain list above, for appending to outbound search queries.

    Returns a single space-joined string, e.g. "-site:indeed.com
    -site:glassdoor.com ...", ready to concatenate onto a query.
    """
    return " ".join(f"-site:{d}" for d in _QUERY_EXCLUSION_DOMAINS)


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
    "in",   # linkedin.com/in/{person} — personal profiles, not companies
    # Job-posting / career pages — explicitly called out in audit findings
    "career", "careers", "hiring", "vacancy", "vacancies",
    "opportunity", "opportunities", "recruitment",
})

# A small, unambiguous set of keywords worth checking as a SUBSTRING at
# ANY depth in the path — not an exact segment match. Real-world job and
# blog URLs almost never use these words as a clean, isolated path
# segment; they show up fused with other words via hyphens or camelCase,
# which an exact "segment == 'jobs'" check misses entirely. Confirmed by
# production evidence — all of the following slipped past the previous
# exact-match version:
#   jobs.insightglobal.com/find_a_job/california/job-539555
#   appleone.com/CareerSearch/JobDetail/Manufacturing-Operations-Manager...
#   careers.ppg.com/us/en/job/PIIPIIUSJR265292EXTERNALENUS/...
#   spacecrew.com/space-jobs/mg68kpnn-northrop-grumman-...
#   phreesia.com/blog-what-is-patient-intake
#   medozai.com/blogs/manual-patient-intake-cost-ai-healthcare
#   jobs.loves.com/inventory-management-coordinator-retail/job/483783
#   careers.stellantis.com/job/23127456/retail-inventory-management-...
#   careers.sysco.com/en/job/houston/manager-supply-chain-...
#   careers.basspro.com/us/en/job/R253751/Inventory-Analyst-I
# "job" alone (not "jobs") is used as the keyword since substring
# matching means "job" already matches "jobs", "job-539555", "jobdetail",
# etc. Same for "career" matching "careers"/"careersearch". "blog"
# matches "blog", "blogs", and "blog-what-is-...".
# False-positive risk is low and was checked directly against known-good
# results from the same production run: coffeymodica.com's own press-
# release path ("coffey-modica-triples-law-office-footprint-...") and
# duanemorris.com's office page ("/offices/newyork.html") contain none
# of these three keywords and continue to pass.
_DEEP_SCAN_KEYWORDS = ("job", "career", "blog")


def _fails_path_check(url: str) -> bool:
    """
    Reject URLs whose path indicates they are not a company's homepage.

    Checks the FIRST path segment after the domain for the full rejection
    set — so example.com/blog/dental-tips is rejected (first segment =
    'blog') but example.com/dental-services is NOT rejected (no rejected
    segment). Additionally scans the FULL path (any depth, substring
    match) for a small high-confidence job/career/blog keyword set, since
    those reliably indicate a job posting or blog post wherever — and
    however — they appear in the URL (e.g. "/find_a_job/", "/space-jobs/",
    "/CareerSearch/", "/blog-what-is-x").
    """
    try:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        path = parsed.path.strip("/").lower()
    except Exception:
        return False

    if not path:
        return False

    segments = path.split("/")
    if segments[0] in _REJECTED_PATH_SEGMENTS:
        return True
    return any(kw in path for kw in _DEEP_SCAN_KEYWORDS)


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

# Job-posting title patterns — "Jobs at Acme", "Acme is hiring", "Now
# Hiring: ...", "3 Open Positions at Acme". These are job-listing pages
# that slip past the domain/path checks when hosted on an otherwise
# uncategorized job-board domain.
_JOB_TITLE_PATTERNS = re.compile(
    r"\b("
    r"jobs?\s+at|"
    r"careers?\s+at|"
    r"is\s+hiring|"
    r"now\s+hiring|"
    r"we(?:'re|\s+are)\s+hiring|"
    r"job\s+opening|"
    r"open\s+positions?|"
    r"apply\s+now|"
    r"vacanc(?:y|ies)"
    r")\b",
    re.IGNORECASE,
)

# Generic content/article/listicle title patterns — blog posts and
# educational articles ("The Ultimate Guide to X", "10 Tips for Y",
# "Patient Intake: Everything You Need to Know") that describe a topic
# rather than name a business, even when hosted on a domain not already
# in the rejected-domain list.
_ARTICLE_TITLE_PATTERNS = re.compile(
    r"(?:"
    r"^\d+\s+(?:tips|ways|reasons|things|steps|mistakes|questions)\b|"
    r"\bultimate\s+guide\b|"
    r"\bguide\s+to\b|"
    r"\btips\s+for\b|"
    r"\bhow\s+to\b|"
    r"\bwhat\s+is\b|"
    r"\beverything\s+you\s+need\s+to\s+know\b|"
    r"\bchecklist\b|"
    r"\bvs\.?\s+\w+:|"
    r"\bnews\b|"
    r"\bmagazine\b"
    r")",
    re.IGNORECASE,
)


def _fails_title_check(title: str) -> bool:
    """Reject result titles that indicate non-commercial entities, job
    postings, or generic content/article pages rather than a business."""
    return bool(
        _REJECTED_TITLE_PATTERNS.search(title)
        or _JOB_TITLE_PATTERNS.search(title)
        or _ARTICLE_TITLE_PATTERNS.search(title)
    )


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