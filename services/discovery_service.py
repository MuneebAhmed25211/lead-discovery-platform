"""
services/discovery_service.py
──────────────────────────────
Discovery Engine orchestration — post-audit refactor.

Changes from original implementation:
  - Integrated four-pass company validation (utils/validator.py)
  - UUIDs generated BEFORE DB insert so id is never empty
  - LinkedIn company page URLs extracted as enrichment, not rejected
  - _to_candidate() validates BEFORE building the object, not after
  - _persist() uses pre-generated IDs in upsert rows
  - search_id lifecycle protected — generated server-side in router

Pipeline:
  Load ICP queries
    → Run each through SearchProviderRouter (Serper → Tavily fallback)
    → For each SearchResult:
        Pass 1: TLD check (.gov/.edu rejected)
        Pass 2: Domain check (directories/aggregators rejected)
        Pass 3: URL path check (/appointment /directory etc. rejected)
        Pass 4: Title check (Association/Foundation/University rejected)
        LinkedIn: extract company page URL as enrichment if present
    → Deduplicate by root domain
    → Assign UUIDs pre-insert
    → Upsert to discovered_companies
    → Return DiscoveryResponse
"""

from __future__ import annotations

import uuid as uuid_lib
from datetime import datetime, timezone

from supabase import Client

from cache.supabase_cache import SupabaseCache
from core.errors import NotFoundError
from core.logging import get_logger
from providers.router import SearchProviderRouter
from schemas.discovery import (
    DiscoveredCompany,
    DiscoveryResponse,
    ProviderSearchOutcome,
    SearchResult,
)
from utils.dedup import extract_domain, root_domain
from utils.validator import validate_candidate

logger = get_logger(__name__)

_MAX_QUERIES_PER_DISCOVERY = 5
_RESULTS_PER_QUERY = 10


class DiscoveryService:

    def __init__(
        self,
        db: Client,
        cache: SupabaseCache,
        router: SearchProviderRouter | None = None,
    ) -> None:
        self._db = db
        self._cache = cache
        self._router = router or SearchProviderRouter(cache=cache)

    async def discover(self, search_id: str, icp_id: str) -> DiscoveryResponse:
        """
        Run the full discovery pipeline for a given ICP.

        Always returns a DiscoveryResponse. Never raises for provider failures.
        Only raises NotFoundError if icp_id doesn't exist (genuine 404).
        """
        search_queries = self._load_icp_queries(icp_id)

        outcomes: list[ProviderSearchOutcome] = []
        for query in search_queries[:_MAX_QUERIES_PER_DISCOVERY]:
            outcome = await self._router.search(query, num_results=_RESULTS_PER_QUERY)
            outcomes.append(outcome)

        candidates = self._validate_and_dedupe(
            outcomes=outcomes,
            search_id=search_id,
            icp_id=icp_id,
        )

        persisted = self._persist(candidates)

        queries_failed = sum(1 for o in outcomes if not o.succeeded)
        queries_executed = len(outcomes) - queries_failed
        partial_failure = queries_failed > 0 and len(persisted) == 0

        logger.info(
            "discovery_complete",
            search_id=search_id,
            icp_id=icp_id,
            total_found=len(persisted),
            queries_executed=queries_executed,
            queries_failed=queries_failed,
        )

        return DiscoveryResponse(
            search_id=search_id,
            companies=persisted,
            total_found=len(persisted),
            queries_executed=queries_executed,
            queries_failed=queries_failed,
            partial_failure=partial_failure,
        )

    # ── ICP loading ──────────────────────────────────────────────────────────

    def _load_icp_queries(self, icp_id: str) -> list[str]:
        try:
            result = (
                self._db.table("icps")
                .select("search_queries")
                .eq("id", icp_id)
                .single()
                .execute()
            )
        except Exception as exc:
            logger.warning("icp_lookup_failed", icp_id=icp_id, error=str(exc))
            raise NotFoundError(resource="ICP", identifier=icp_id)

        if not result.data:
            raise NotFoundError(resource="ICP", identifier=icp_id)

        queries = result.data.get("search_queries", [])
        if not queries:
            logger.warning("icp_has_no_queries", icp_id=icp_id)
            return []

        return queries

    # ── Validation + deduplication ───────────────────────────────────────────

    def _validate_and_dedupe(
        self,
        outcomes: list[ProviderSearchOutcome],
        search_id: str,
        icp_id: str,
    ) -> list[DiscoveredCompany]:
        """
        Validate each result through four passes and deduplicate by root domain.

        Validation happens BEFORE building DiscoveredCompany objects — nothing
        invalid ever gets constructed. UUIDs are assigned here, pre-insert,
        so they're guaranteed non-empty regardless of DB outcome.
        """
        seen: dict[str, DiscoveredCompany] = {}
        rejection_counts: dict[str, int] = {}

        for outcome in outcomes:
            if not outcome.succeeded:
                continue

            for result in outcome.results:
                candidate = self._validate_and_build(
                    result=result,
                    matched_query=outcome.query,
                    search_id=search_id,
                    icp_id=icp_id,
                    rejection_counts=rejection_counts,
                )
                if candidate is None:
                    continue

                existing = seen.get(candidate.domain)
                if existing is None:
                    seen[candidate.domain] = candidate
                else:
                    # Accumulate matched queries — multiple ICP queries
                    # finding the same domain is a stronger signal.
                    if candidate.matched_query not in existing.matched_query:
                        existing.matched_query = (
                            f"{existing.matched_query} | {candidate.matched_query}"
                        )
                    # Enrich with LinkedIn URL if not already set
                    if candidate.linkedin_url and not existing.linkedin_url:
                        existing.linkedin_url = candidate.linkedin_url

        if rejection_counts:
            logger.info(
                "discovery_validation_summary",
                search_id=search_id,
                rejections=rejection_counts,
                accepted=len(seen),
            )

        return list(seen.values())

    def _validate_and_build(
        self,
        result: SearchResult,
        matched_query: str,
        search_id: str,
        icp_id: str,
        rejection_counts: dict[str, int],
    ) -> DiscoveredCompany | None:
        """
        Run the four-pass validation and build a DiscoveredCompany on success.

        Returns None for rejected results. UUID is assigned here so it exists
        on the object before any DB call.
        """
        raw_domain = extract_domain(result.url)
        domain = root_domain(raw_domain)

        if not domain or "." not in domain:
            rejection_counts["malformed_domain"] = rejection_counts.get("malformed_domain", 0) + 1
            return None

        validation = validate_candidate(
            url=result.url,
            title=result.title,
            domain=domain,
        )

        if not validation.passed:
            reason_key = validation.rejection_reason.split(":")[0]
            rejection_counts[reason_key] = rejection_counts.get(reason_key, 0) + 1
            logger.debug(
                "candidate_rejected",
                domain=domain,
                reason=validation.rejection_reason,
                title=result.title[:60],
            )
            return None

        return DiscoveredCompany(
            id=str(uuid_lib.uuid4()),   # assigned here — never empty
            search_id=search_id,
            icp_id=icp_id,
            name=_derive_name(result.title),
            domain=domain,
            website=f"https://{domain}",
            source_url=result.url,
            snippet=result.snippet,
            provider=result.provider,
            matched_query=matched_query,
            linkedin_url=validation.linkedin_url,
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, candidates: list[DiscoveredCompany]) -> list[DiscoveredCompany]:
        """
        Upsert candidates into discovered_companies.

        IDs are pre-generated on the DiscoveredCompany objects, so even if
        the DB upsert fails and we fall back to returning in-memory candidates,
        every returned object has a real, non-empty UUID.
        """
        if not candidates:
            return []

        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "id": c.id,              # pre-generated UUID — never empty
                "search_id": c.search_id,
                "icp_id": c.icp_id,
                "name": c.name,
                "domain": c.domain,
                "website": c.website,
                "source_url": c.source_url,
                "snippet": c.snippet,
                "provider": c.provider,
                "matched_query": c.matched_query,
                "linkedin_url": c.linkedin_url,
                "discovered_at": now,
            }
            for c in candidates
        ]

        try:
            result = (
                self._db.table("discovered_companies")
                .upsert(rows, on_conflict="search_id,domain")
                .execute()
            )
        except Exception as exc:
            logger.error(
                "discovery_persist_failed",
                error=str(exc),
                candidate_count=len(candidates),
            )
            # In-memory candidates already have real UUIDs — safe to return
            return candidates

        # Build return list from DB response rows; fall back to in-memory
        # candidates (which already have pre-generated IDs) if DB returned nothing.
        if result.data:
            persisted: list[DiscoveredCompany] = []
            for row in result.data:
                persisted.append(
                    DiscoveredCompany(
                        id=row["id"],
                        search_id=row["search_id"],
                        icp_id=row["icp_id"],
                        name=row["name"],
                        domain=row["domain"],
                        website=row["website"],
                        source_url=row["source_url"],
                        snippet=row["snippet"],
                        provider=row["provider"],
                        matched_query=row["matched_query"],
                        linkedin_url=row.get("linkedin_url", ""),
                        discovered_at=row["discovered_at"],
                    )
                )
            return persisted

        # DB returned no rows (can happen with certain Supabase upsert configs)
        # — return in-memory candidates, which already have real pre-generated IDs.
        logger.warning(
            "discovery_persist_no_rows_returned",
            candidate_count=len(candidates),
        )
        return candidates


# ── Helpers ───────────────────────────────────────────────────────────────────

def _derive_name(title: str) -> str:
    """
    Extract a company name from a search result title.

    Search result titles commonly have suffixes like " | Home", " - Official
    Site", " :: Services". Strip at common separators and take the first
    segment, which is usually the brand name.
    """
    for separator in (" | ", " - ", " — ", " :: ", " · ", " > "):
        if separator in title:
            return title.split(separator)[0].strip()
    return title.strip()
