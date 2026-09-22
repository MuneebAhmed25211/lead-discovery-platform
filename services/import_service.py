"""
services/import_service.py
────────────────────────────
Path 2: CSV import. Lets a user bring their own list of companies instead
of Discovery generating one from ICP search queries.

Deliberately reuses as much of Discovery's machinery as possible rather
than building a parallel pipeline:
  - Same DiscoveredCompany output shape → Research/Score/Qualify and every
    result-page component work completely unchanged downstream.
  - Same validate_candidate() checks → an imported row that resolves to a
    magazine, directory, government body, etc. gets caught the exact same
    way a search-discovered one would, not a weaker or different check.
  - Same DiscoveryService._persist() (via the public persist_companies
    wrapper) → one upsert code path regardless of where a company came
    from.
  - Same SearchProviderRouter (Serper→Tavily fallback, same cache) → a
    name-only row's resolution search behaves identically to a normal
    discovery query, including caching and fallback behavior.

What's actually new: parsing a CSV, and resolving a bare company name to a
website when the CSV didn't already supply one.

The 15-row cap (_MAX_IMPORT_ROWS in schemas/discovery.py) is a deliberate
demo limit, not a technical one — surfaced to the user via
CsvParseResponse BEFORE any search/crawl/Groq cost is spent, with the
messaging framing it as a natural upgrade path rather than a hidden
restriction (see routers/import_csv.py's docstring for the exact copy
agreed with the user).
"""

from __future__ import annotations

import csv
import io
import uuid as uuid_lib

from supabase import Client

from cache.supabase_cache import SupabaseCache
from core.logging import get_logger
from providers.router import SearchProviderRouter
from schemas.discovery import (
    CsvImportResponse,
    DiscoveredCompany,
    ImportedLeadRow,
)
from services.discovery_service import DiscoveryService, _derive_name
from utils.dedup import extract_domain, root_domain
from utils.validator import validate_candidate

logger = get_logger(__name__)

_MAX_IMPORT_ROWS = 15

# Column-name aliases we'll recognize in an uploaded CSV's header row,
# checked case-insensitively. Real user CSVs vary a lot (a LinkedIn Sales
# Navigator export, a trade-show badge-scan list, a manually-kept sheet),
# so this is deliberately lenient rather than requiring one exact schema.
_NAME_COLUMN_ALIASES = {"company", "company name", "name", "business", "business name", "organization"}
_WEBSITE_COLUMN_ALIASES = {
    "website", "domain", "url", "site", "company website", "web",
    # Confirmed real gap: Apollo's own CSV export uses "website_url" and
    # "primary_domain" as column headers, matching neither of the
    # original aliases above. Because Apollo's export DOES have a
    # recognized name column ("name"), the old code never fell back to
    # positional guessing either — every row silently got raw_website=""
    # despite the domain being right there in the file, forcing every
    # row through web-search resolution instead of using the CSV's own
    # data. Confirmed via a real Apollo export row: header
    # "...,name,primary_domain,website_url,..." → a row for "Nexie" with
    # primary_domain "nexie.ai" got resolved via search to an unrelated
    # "nexie.com" with a garbled merged name.
    "website_url", "primary_domain", "company_domain",
}


class ImportService:

    def __init__(
        self,
        db: Client,
        cache: SupabaseCache,
        router: SearchProviderRouter | None = None,
    ) -> None:
        self._db = db
        self._cache = cache
        self._router = router or SearchProviderRouter(cache=cache)
        # Reused for its persist_companies() and _validate_and_build()-style
        # logic — see module docstring for why this composes rather than
        # duplicates Discovery's pipeline.
        self._discovery = DiscoveryService(db=db, cache=cache, router=self._router)

    # ── CSV parsing (no cost — no search/crawl/Groq calls happen here) ───────

    @staticmethod
    def parse_csv(file_bytes: bytes) -> list[ImportedLeadRow]:
        """
        Parse an uploaded CSV into ImportedLeadRow objects.

        Lenient column matching: looks for a header matching one of the
        name/website alias sets above; if no header matches at all (e.g.
        a headerless two-column sheet), falls back to "first column is
        the name, second column (if present) is the website." Blank name
        cells are skipped rather than raising — a user's real-world sheet
        will often have stray blank rows.
        """
        text = file_bytes.decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(text))
        raw_rows = [r for r in reader if any(cell.strip() for cell in r)]
        if not raw_rows:
            return []

        header = [cell.strip().lower() for cell in raw_rows[0]]
        name_idx = next((i for i, h in enumerate(header) if h in _NAME_COLUMN_ALIASES), None)
        website_idx = next((i for i, h in enumerate(header) if h in _WEBSITE_COLUMN_ALIASES), None)

        if name_idx is not None:
            data_rows = raw_rows[1:]
        else:
            # No recognized header — treat every row (including the
            # first) as data, positionally: column 0 = name, column 1 =
            # website if present.
            name_idx = 0
            website_idx = 1 if len(header) > 1 else None
            data_rows = raw_rows

        rows: list[ImportedLeadRow] = []
        for raw in data_rows:
            if name_idx >= len(raw):
                continue
            name = raw[name_idx].strip()
            if not name:
                continue
            website = raw[website_idx].strip() if website_idx is not None and website_idx < len(raw) else ""
            rows.append(ImportedLeadRow(raw_name=name, raw_website=website))

        return rows

    # ── Resolution + validation + persistence ────────────────────────────────

    async def resolve_and_import(
        self, rows: list[ImportedLeadRow], search_id: str, icp_id: str
    ) -> CsvImportResponse:
        """
        Resolve each row to a validated website (skipping the search step
        for rows that already supplied one), dedupe by domain, persist,
        and return the same shape Discovery returns.

        The cap is re-enforced HERE regardless of how many rows the
        caller sent — see schemas/discovery.py's CsvImportRequest
        docstring for why the client is never trusted for this.
        """
        capped_rows = rows[:_MAX_IMPORT_ROWS]

        seen: dict[str, DiscoveredCompany] = {}
        unresolved: list[str] = []

        for row in capped_rows:
            candidate = (
                self._build_from_supplied_website(row, search_id, icp_id)
                if row.raw_website
                else await self._resolve_by_search(row, search_id, icp_id)
            )
            if candidate is None:
                unresolved.append(row.raw_name)
                continue
            if candidate.domain not in seen:
                seen[candidate.domain] = candidate

        persisted = self._discovery.persist_companies(list(seen.values()))

        logger.info(
            "csv_import_complete",
            search_id=search_id,
            icp_id=icp_id,
            rows_submitted=len(rows),
            rows_processed=len(capped_rows),
            resolved=len(persisted),
            unresolved=len(unresolved),
        )

        return CsvImportResponse(
            search_id=search_id,
            companies=persisted,
            total_found=len(persisted),
            rows_submitted=len(rows),
            rows_processed=len(capped_rows),
            unresolved=unresolved,
        )

    def _build_from_supplied_website(
        self, row: ImportedLeadRow, search_id: str, icp_id: str
    ) -> DiscoveredCompany | None:
        """
        A row that already has a website: validate it the same way a
        discovered candidate would be, just without real search-snippet
        text to check (there wasn't a search — the user supplied this
        directly). Pass 1-3 (TLD/domain/path, which work off the URL
        alone) still apply in full; the title/content-based checks use
        the CSV's own company name as "title" with an empty snippet —
        exactly the same reduced-evidence shape research_service.py's
        post-crawl revalidation already uses, so this isn't a new pattern,
        just an earlier use of an existing one. Any genuine noise that
        evidence-starved check can't catch yet still gets a second chance
        at Research's post-crawl revalidation once the site is actually
        crawled.
        """
        raw_domain = extract_domain(row.raw_website)
        domain = root_domain(raw_domain)
        if not domain or "." not in domain:
            return None

        url = row.raw_website if row.raw_website.startswith(("http://", "https://")) else f"https://{domain}"

        validation = validate_candidate(url=url, title=row.raw_name, domain=domain, snippet="")
        if not validation.passed:
            logger.debug("csv_row_rejected", domain=domain, reason=validation.rejection_reason)
            return None

        return DiscoveredCompany(
            id=str(uuid_lib.uuid4()),
            search_id=search_id,
            icp_id=icp_id,
            name=row.raw_name,
            domain=domain,
            website=f"https://{domain}",
            source_url=url,
            snippet="",
            provider="csv_import",
            matched_query=f"csv_import:{row.raw_name}",
            linkedin_url=validation.linkedin_url,
        )

    async def _resolve_by_search(
        self, row: ImportedLeadRow, search_id: str, icp_id: str
    ) -> DiscoveredCompany | None:
        """
        A row with no website: one targeted search to find the company's
        official site, through the exact same Serper→Tavily router (and
        cache) Discovery uses for its own queries. Takes the first result
        that passes the full validate_candidate() check — this is the one
        place a name-only import gets the full benefit of real
        search-snippet text, the same evidence a normal discovery result
        would have.
        """
        query = f'"{row.raw_name}" official website'
        outcome = await self._router.search(query, num_results=5)
        if not outcome.succeeded:
            return None

        for result in outcome.results:
            raw_domain = extract_domain(result.url)
            domain = root_domain(raw_domain)
            if not domain or "." not in domain:
                continue

            validation = validate_candidate(
                url=result.url, title=result.title, domain=domain, snippet=result.snippet
            )
            if not validation.passed:
                continue

            return DiscoveredCompany(
                id=str(uuid_lib.uuid4()),
                search_id=search_id,
                icp_id=icp_id,
                name=_derive_name(result.title) or row.raw_name,
                domain=domain,
                website=f"https://{domain}",
                source_url=result.url,
                snippet=result.snippet,
                provider="csv_import",
                matched_query=f"csv_import:{row.raw_name}",
                linkedin_url=validation.linkedin_url,
            )

        return None