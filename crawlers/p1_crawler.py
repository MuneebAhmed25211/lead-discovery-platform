"""
crawlers/p1_crawler.py
───────────────────────
Async P1 (company's own website) page fetcher.

Responsibilities:
  - Resolve which pages to crawl for a given domain
  - Respect robots.txt (fetch once per domain, cache result in memory)
  - Fetch each page with timeout + single retry
  - Return structured CrawlResult objects consumed by extractors
  - Never raise — failed pages are recorded in CrawlResult.error

Concurrency:
  - Uses asyncio.Semaphore to cap simultaneous open connections
  - Default: 5 concurrent requests across all companies in a batch
  - Per-company: pages are fetched concurrently (homepage + about + etc.)

Page priority order (most signal-rich first):
  1. /          homepage
  2. /about     or /about-us
  3. /services  or /our-services
  4. /team      or /leadership  or /our-team
  5. /careers   or /jobs
  6. /contact   or /contact-us

For each candidate path we try the most common variant first. If it
returns a non-200, we try the alternate slug. If both fail, the page
is skipped and recorded in CrawlResult.pages_failed.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Semaphore shared across all crawl calls in a process — caps total open sockets
_GLOBAL_SEMAPHORE = asyncio.Semaphore(5)

# Cap redirect chains — open-ended redirect following is both a minor
# DoS surface and a source of slow, wasted crawl time on misconfigured sites
_MAX_REDIRECTS = 5

# Connection pool limits — reused when a shared client is provided for
# batch crawling, so 1,000+ companies in a batch don't each pay full
# TCP/TLS handshake cost. Keepalive connections are capped to avoid
# unbounded socket growth at the 10,000-100,000 company scale.
_POOL_LIMITS = httpx.Limits(
    max_connections=50,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
)

# Page slug candidates in priority order: (primary_slug, *alternates)
_PAGE_CANDIDATES: list[tuple[str, ...]] = [
    ("/",),
    ("/about", "/about-us", "/about-us/", "/who-we-are", "/our-story"),
    ("/services", "/our-services", "/what-we-do", "/solutions", "/offerings"),
    ("/team", "/our-team", "/leadership", "/about/team", "/about/leadership"),
    ("/careers", "/jobs", "/join-us", "/work-with-us", "/join-our-team"),
    ("/contact", "/contact-us", "/get-in-touch", "/reach-us"),
]

# httpx headers — identify the bot politely
_HEADERS = {
    "User-Agent": settings.CRAWLER_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PageResult:
    """The raw result of fetching a single URL."""

    url: str
    status_code: int = 0
    html: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and bool(self.html)


@dataclass
class CrawlResult:
    """All pages fetched for a single company domain."""

    domain: str
    website: str
    pages: list[PageResult] = field(default_factory=list)
    pages_crawled: list[str] = field(default_factory=list)
    pages_failed: list[str] = field(default_factory=list)
    robots_blocked: bool = False
    error: str = ""

    @property
    def successful_pages(self) -> list[PageResult]:
        return [p for p in self.pages if p.ok]

    @property
    def ok(self) -> bool:
        return len(self.successful_pages) > 0


# ── Robots.txt cache (in-process, not persisted) ──────────────────────────────

_robots_cache: dict[str, RobotFileParser] = {}


async def _fetch_robots(domain: str, client: httpx.AsyncClient) -> RobotFileParser:
    """
    Fetch and parse robots.txt for a domain. Cached per process.

    IMPORTANT: an unparsed RobotFileParser defaults can_fetch() to False,
    not True — the opposite of "no robots.txt = open". When robots.txt is
    missing (404) or unreachable, we must explicitly load an allow-all
    ruleset rather than leaving the parser in its empty default state.
    """
    if domain in _robots_cache:
        return _robots_cache[domain]

    rp = RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"
    try:
        resp = await client.get(robots_url, timeout=5)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            # No robots.txt (404, 403, etc.) → explicitly allow all.
            # An empty/unparsed RobotFileParser defaults to DISALLOW,
            # which is the wrong default for "robots.txt doesn't exist".
            rp.parse(["User-agent: *", "Allow: /"])
    except Exception:
        # Network error fetching robots.txt → explicitly allow all,
        # same reasoning as above.
        rp.parse(["User-agent: *", "Allow: /"])

    _robots_cache[domain] = rp
    return rp


def _is_allowed(rp: RobotFileParser, url: str) -> bool:
    """Return True if our user agent is allowed to fetch this URL."""
    if not settings.CRAWLER_RESPECT_ROBOTS:
        return True
    try:
        return rp.can_fetch(settings.CRAWLER_USER_AGENT, url)
    except Exception:
        return True  # parse error → allow


# ── Page fetcher ──────────────────────────────────────────────────────────────

async def _fetch_page(
    client: httpx.AsyncClient,
    url: str,
    timeout: int,
) -> PageResult:
    """
    Fetch a single URL. Single retry on connection error only.
    Returns PageResult regardless of outcome — never raises.
    """
    for attempt in range(2):
        try:
            async with _GLOBAL_SEMAPHORE:
                resp = await client.get(
                    url,
                    timeout=timeout,
                    follow_redirects=True,
                    headers=_HEADERS,
                )
            if resp.status_code == 200:
                return PageResult(url=url, status_code=200, html=resp.text)
            return PageResult(url=url, status_code=resp.status_code)
        except httpx.TimeoutException:
            return PageResult(url=url, error=f"timeout after {timeout}s")
        except httpx.TooManyRedirects:
            return PageResult(url=url, error="too_many_redirects")
        except httpx.HTTPError as exc:
            if attempt == 0:
                await asyncio.sleep(1)
                continue
            return PageResult(url=url, error=str(exc)[:120])
        except Exception as exc:
            return PageResult(url=url, error=f"unexpected:{str(exc)[:80]}")

    return PageResult(url=url, error="exhausted_retries")


async def _resolve_page(
    client: httpx.AsyncClient,
    base_url: str,
    slugs: tuple[str, ...],
    rp: RobotFileParser,
    timeout: int,
) -> PageResult:
    """
    Try each slug in order, return the first successful PageResult.

    If every slug fails (non-200, network error, or robots-blocked),
    returns a PageResult representing the PRIMARY slug (slugs[0]) marked
    as failed, so the caller can accurately record which page-type could
    not be retrieved rather than silently dropping the attempt.
    """
    primary_url = urljoin(base_url, slugs[0])
    last_status = 0
    last_error = ""
    any_robots_blocked = False

    for slug in slugs:
        url = urljoin(base_url, slug)
        if not _is_allowed(rp, url):
            logger.debug("robots_blocked", url=url)
            any_robots_blocked = True
            continue
        result = await _fetch_page(client, url, timeout)
        if result.ok:
            return result
        # Track the most informative failure for reporting
        if result.status_code:
            last_status = result.status_code
        if result.error:
            last_error = result.error

    # All slugs failed — return a failed PageResult for the primary slug
    # so pages_failed accurately reflects which page-type was attempted.
    if last_error:
        return PageResult(url=primary_url, error=last_error)
    if last_status:
        return PageResult(url=primary_url, status_code=last_status)
    if any_robots_blocked:
        return PageResult(url=primary_url, error="robots_disallowed")
    return PageResult(url=primary_url, error="no_slug_resolved")


# ── Shared client factory (for batch crawling connection pool reuse) ─────────

def make_client(timeout: int | None = None) -> httpx.AsyncClient:
    """
    Build an httpx.AsyncClient configured with sane production defaults:
    bounded redirect following, connection pool limits, and our identifying
    headers.

    Use this directly (with manual close) or via crawl_client() as an
    async context manager when crawling many companies in a batch, so
    TCP/TLS connections are reused across companies rather than rebuilt
    on every single crawl_company() call.
    """
    return httpx.AsyncClient(
        timeout=timeout or settings.CRAWLER_TIMEOUT_SECONDS,
        follow_redirects=True,
        max_redirects=_MAX_REDIRECTS,
        limits=_POOL_LIMITS,
        headers=_HEADERS,
    )


@asynccontextmanager
async def crawl_client(timeout: int | None = None):
    """
    Async context manager yielding a shared, pooled httpx.AsyncClient.

    Usage in batch research:
        async with crawl_client() as client:
            for company in companies:
                result = await crawl_company(domain, website, client=client)

    This is the recommended path at scale (hundreds to tens of thousands
    of companies) — it avoids a fresh TCP/TLS handshake per company.
    """
    client = make_client(timeout)
    try:
        yield client
    finally:
        await client.aclose()


# ── Public API ────────────────────────────────────────────────────────────────

async def crawl_company(
    domain: str,
    website: str,
    client: httpx.AsyncClient | None = None,
) -> CrawlResult:
    """
    Crawl all P1 pages for a company domain.

    Args:
        domain:  Root domain, e.g. "acme.com"
        website: Full homepage URL, e.g. "https://acme.com"
        client:  Optional shared httpx.AsyncClient for connection pool
                 reuse across a batch. If omitted, a dedicated client is
                 created and closed for this single call — fine for
                 one-off research but wasteful at batch scale.

    Returns:
        CrawlResult with all successfully fetched pages.
        Never raises — errors are captured in CrawlResult.error.
    """
    result = CrawlResult(domain=domain, website=website)
    timeout = settings.CRAWLER_TIMEOUT_SECONDS

    # Normalise base URL
    if not website.startswith(("http://", "https://")):
        website = f"https://{website}"

    try:
        if client is not None:
            # Reuse the caller's pooled client — no new connection setup
            rp = await _fetch_robots(domain, client)
            await _crawl_with_client(client, domain, website, rp, timeout, result)
        else:
            # Standalone call — own client, closed when done
            async with make_client(timeout) as owned_client:
                rp = await _fetch_robots(domain, owned_client)
                await _crawl_with_client(owned_client, domain, website, rp, timeout, result)

    except Exception as exc:
        result.error = str(exc)[:200]
        logger.warning("crawl_company_failed", domain=domain, error=result.error)

    logger.info(
        "crawl_complete",
        domain=domain,
        pages_ok=len(result.pages_crawled),
        pages_failed=len(result.pages_failed),
        robots_blocked=result.robots_blocked,
    )
    return result


async def _crawl_with_client(
    client: httpx.AsyncClient,
    domain: str,
    website: str,
    rp: RobotFileParser,
    timeout: int,
    result: CrawlResult,
) -> None:
    """Shared crawl body used by both the pooled and standalone code paths."""
    # Check if the homepage itself is blocked
    if not _is_allowed(rp, website):
        result.robots_blocked = True
        logger.info("crawl_robots_blocked_homepage", domain=domain)
        return

    # Fetch all candidate pages concurrently
    tasks = [
        _resolve_page(client, website, slugs, rp, timeout)
        for slugs in _PAGE_CANDIDATES
    ]
    page_results = await asyncio.gather(*tasks, return_exceptions=True)

    for page in page_results:
        if isinstance(page, Exception):
            continue
        if page.ok:
            result.pages.append(page)
            result.pages_crawled.append(page.url)
        else:
            # _resolve_page always returns a PageResult representing
            # the primary slug for this page-type, even when every
            # variant failed — so this list accurately reflects
            # which page-types could not be retrieved.
            result.pages_failed.append(page.url)
