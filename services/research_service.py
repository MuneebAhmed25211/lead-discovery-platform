"""
services/research_service.py
─────────────────────────────
Research Engine — Phase 4 orchestrator.

For a single discovered company:
  1. Crawl P1 pages (crawlers/p1_crawler.py)
  2. Run all 9 extractors against the crawl result
  3. Collect ExtractedFact list from every extractor
  4. Write evidence rows to Supabase
  5. Derive the companies row from the collected evidence
  6. Set research_status = 'complete' | 'failed' | 'skipped'
  7. Update discovered_companies.promoted_to_company_id

Zero Groq calls. Zero LLM usage. All facts are deterministic.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup
from supabase import Client

from core.errors import NotFoundError
from core.logging import get_logger
from crawlers.extractors.base import (
    ExtractedFact,
    F_BOOKING_WIDGET,
    F_CONTACT_EMAIL,
    F_CONTACT_PAGE,
    F_CONTACT_PHONE,
    F_EMPLOYEE_RANGE,
    F_HIRING_SIGNAL,
    F_INDUSTRY,
    F_JOB_TITLE,
    F_LOCATION,
    F_NAME,
    F_SERVICE,
    F_SOCIAL_FACEBOOK,
    F_SOCIAL_INSTAGRAM,
    F_SOCIAL_LINKEDIN,
    F_SOCIAL_TWITTER,
    F_TECHNOLOGY,
)
from crawlers.extractors.booking import extract_booking
from crawlers.extractors.contact import extract_contact
from crawlers.extractors.employee_count import extract_employee_count
from crawlers.extractors.hiring import extract_hiring
from crawlers.extractors.industry import extract_industry
from crawlers.extractors.location import extract_location
from crawlers.extractors.name import extract_name
from crawlers.extractors.services import extract_services
from crawlers.extractors.social import extract_social
from crawlers.extractors.technologies import extract_technologies
from crawlers.p1_crawler import CrawlResult, PageResult, crawl_client, crawl_company
from schemas.company import Company, ResearchResponse, ResearchStatus
from utils.validator import validate_candidate
from utils.wikidata_check import check_wikidata_entity_type

logger = get_logger(__name__)

_BATCH_CONCURRENCY = 3


class ResearchService:

    # Field-importance weights for the overall company confidence score.
    # A plain average (the previous implementation) treats a 0.55-confidence
    # phone-number match exactly the same as a 0.90-confidence industry
    # classification, so a company with lots of low-value contact/social
    # facts can outscore one with a few precise core facts. Core identity
    # and firmographic fields drive Phase 5 scoring, so they carry the
    # most weight here; contact/social facts are still useful signal but
    # shouldn't dominate the confidence score.
    _FIELD_WEIGHT: dict[str, float] = {
        F_NAME: 2.0,
        F_INDUSTRY: 2.0,
        F_LOCATION: 1.5,
        F_EMPLOYEE_RANGE: 1.5,
        F_SERVICE: 1.0,
        F_TECHNOLOGY: 1.0,
        F_HIRING_SIGNAL: 1.0,
        F_BOOKING_WIDGET: 0.7,
        F_JOB_TITLE: 0.5,
        F_CONTACT_EMAIL: 0.5,
        F_CONTACT_PHONE: 0.5,
        F_SOCIAL_LINKEDIN: 0.5,
        F_SOCIAL_TWITTER: 0.3,
        F_SOCIAL_FACEBOOK: 0.3,
        F_SOCIAL_INSTAGRAM: 0.3,
        F_CONTACT_PAGE: 0.3,
    }
    _DEFAULT_FIELD_WEIGHT = 0.5

    def __init__(self, db: Client) -> None:
        self._db = db

    async def research_company(
        self,
        discovered_company_id: str,
        client: httpx.AsyncClient | None = None,
    ) -> ResearchResponse:
        row = self._load_discovered(discovered_company_id)
        if row is None:
            raise NotFoundError(resource="discovered_company", identifier=discovered_company_id)

        domain = row["domain"]
        website = row["website"]
        search_id = row["search_id"]
        discovery_source_url = row.get("source_url", "")
        discovery_snippet = row.get("snippet", "")

        logger.info("research_start", domain=domain)

        crawl_result = await crawl_company(domain=domain, website=website, client=client)

        if crawl_result.robots_blocked:
            return self._handle_skipped(
                discovered_company_id, search_id, domain, website,
                discovery_source_url, discovery_snippet, "robots_blocked",
            )

        if not crawl_result.ok:
            return self._handle_failed(
                discovered_company_id, search_id, domain, website,
                discovery_source_url, discovery_snippet, crawl_result,
            )

        # Everything from here on (extraction → build → persist) is wrapped
        # so that any unexpected error — not just a single extractor
        # throwing — still results in a determinate, persisted
        # research_status instead of an unhandled exception that leaves
        # no company row and no record of what happened to this domain.
        try:
            facts = self._run_extractors(crawl_result.successful_pages)
            facts = self._dedupe_facts(facts)

            if not facts:
                return self._handle_failed(
                    discovered_company_id, search_id, domain, website,
                    discovery_source_url, discovery_snippet, crawl_result,
                )

            # Post-crawl re-validation. Discovery-time validation
            # (utils/validator.validate_candidate, called from
            # discovery_service.py) only ever sees the search provider's
            # indexed title/snippet for a result — not the live page. When
            # a search engine's cached title differs from the site's real
            # <title>/og:site_name (common for content sites that get
            # re-titled per-article, or whose homepage title is shorter
            # than their real masthead), a noise candidate that should
            # have been rejected (a magazine, directory, association,
            # etc.) can slip past discovery validation and only reveal
            # itself once we've crawled the actual page. Confirmed real
            # case: foodandbeverage.business's crawled name is "Food and
            # Beverage Business Magazine" (extract_name.py catches this
            # via og:site_name/<title>, both of which literally contain
            # "Magazine"), but discovery-time validation passed it —
            # meaning the search provider's title for that result didn't
            # contain the word. Re-running the same title/content checks
            # here, against ground-truth crawled data instead of
            # discovery-time metadata, closes that gap. Pass 1-3
            # (URL/domain-based) already ran at discovery time and can't
            # change, but the content-based checks (media publication,
            # directory, association, event, Built In network, etc.) now
            # get a real snippet built from the crawled pages themselves
            # — see _crawled_content_snippet's docstring for why title
            # alone wasn't enough (confirmed real case: Startup Savant's
            # "editorial team" / "Media Kit" self-description lives on
            # its /about-us page, not in its <title>).
            canonical_name = self._best_field(facts, F_NAME)
            crawled_snippet = self._crawled_content_snippet(crawl_result)
            if canonical_name or crawled_snippet:
                post_crawl_check = validate_candidate(
                    url=discovery_source_url or website,
                    title=canonical_name,
                    domain=domain,
                    snippet=crawled_snippet,
                )
                if not post_crawl_check.passed:
                    logger.info(
                        "post_crawl_validation_rejected",
                        domain=domain,
                        canonical_name=canonical_name,
                        reason=post_crawl_check.rejection_reason,
                    )
                    return self._handle_skipped(
                        discovered_company_id, search_id, domain, website,
                        discovery_source_url, discovery_snippet,
                        reason=f"post_crawl_noise:{post_crawl_check.rejection_reason}",
                    )

            # Wikidata entity-type check — a structural upgrade discussed
            # with the user rather than another hand-written regex.
            # utils/validator.py's checks above (and everywhere else in
            # this file) all work the same reactive way: a real case
            # leaks, its own self-description gets read, a regex gets
            # written to catch that exact phrasing. This check instead
            # asks a pre-vetted structured database (Wikidata's community-
            # maintained "instance of" property) whether the candidate is
            # a government body, news outlet, or nonprofit — for any
            # candidate notable enough to have a Wikidata page. Most
            # small private companies (the bulk of any real ICP) won't
            # have one, and that's fine: WikidataCheckResult.checked=False
            # is treated identically to "passed," never as a rejection.
            # Only an explicit, confident, exact-name match against a
            # known disqualifying type ever rejects a candidate here — see
            # utils/wikidata_check.py's module docstring for the full
            # rationale and stated limits.
            if canonical_name:
                wikidata_result = await check_wikidata_entity_type(client, canonical_name)
                if wikidata_result.flagged:
                    logger.info(
                        "wikidata_entity_type_rejected",
                        domain=domain,
                        canonical_name=canonical_name,
                        matched_type=wikidata_result.matched_type_label,
                    )
                    return self._handle_skipped(
                        discovered_company_id, search_id, domain, website,
                        discovery_source_url, discovery_snippet,
                        reason=f"wikidata_entity_type:{wikidata_result.matched_type_label}",
                    )

            company_id = str(uuid.uuid4())
            company = self._build_company(
                company_id, search_id, domain, website,
                discovery_source_url, discovery_snippet, facts, crawl_result,
            )

            self._persist_company(company, company_id)
            self._persist_evidence(company_id, facts)
            self._promote_discovered(discovered_company_id, company_id)

            logger.info(
                "research_complete",
                domain=domain,
                evidence_count=len(facts),
                confidence=company.data_confidence,
            )

            return ResearchResponse(
                company=company,
                evidence_count=len(facts),
                p1_pages_crawled=crawl_result.pages_crawled,
                p1_pages_failed=crawl_result.pages_failed,
                data_confidence=company.data_confidence,
                cached=False,
            )
        except Exception as exc:
            logger.error("research_unexpected_error", domain=domain, error=str(exc))
            return self._handle_failed(
                discovered_company_id, search_id, domain, website,
                discovery_source_url, discovery_snippet, crawl_result,
            )

    async def research_batch(self, search_id: str, max_companies: int = 20) -> dict[str, Any]:
        unpromoted = self._load_unpromoted(search_id, limit=max_companies)
        if not unpromoted:
            return {"researched": 0, "failed": 0, "skipped": 0, "total": 0}

        semaphore = asyncio.Semaphore(_BATCH_CONCURRENCY)
        results: dict[str, int] = {"researched": 0, "failed": 0, "skipped": 0, "total": len(unpromoted)}

        # Share one pooled httpx client across the entire batch — avoids a
        # fresh TCP/TLS handshake per company. At 100-100,000 company scale
        # this is the difference between thousands of connection setups and
        # a small reused pool (see crawlers/p1_crawler.py:_POOL_LIMITS).
        async with crawl_client() as shared_client:

            async def research_one(row: dict) -> None:
                async with semaphore:
                    try:
                        resp = await self.research_company(row["id"], client=shared_client)
                        status = resp.company.research_status
                        if status == ResearchStatus.COMPLETE:
                            results["researched"] += 1
                        elif status == ResearchStatus.SKIPPED:
                            results["skipped"] += 1
                        else:
                            results["failed"] += 1
                    except Exception as exc:
                        logger.error("batch_company_error", domain=row.get("domain"), error=str(exc))
                        results["failed"] += 1

            await asyncio.gather(*[research_one(row) for row in unpromoted])
        logger.info("batch_research_complete", **results)
        return results

    def _run_extractors(self, pages: list[PageResult]) -> list[ExtractedFact]:
        all_facts: list[ExtractedFact] = []
        for extractor in (
            extract_name, extract_location, extract_industry,
            extract_services, extract_employee_count, extract_technologies,
            extract_hiring, extract_contact, extract_social, extract_booking,
        ):
            try:
                all_facts.extend(extractor(pages))
            except Exception as exc:
                logger.warning("extractor_error", extractor=extractor.__name__, error=str(exc))
        return all_facts

    def _dedupe_facts(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        """
        Collapse facts that share the same (field_name, value) — e.g. two
        extractors independently surfacing the same phone number, or the
        same fact appearing verbatim on two crawled pages.

        Each extractor already dedupes its own output internally, but that
        only guards against duplicates *within* one extractor's results.
        This is the safety net across the whole batch: it guarantees the
        evidence table never receives two rows for the same field/value
        combination for a company, regardless of which extractor(s)
        produced them. Keeps the highest-confidence instance of each
        duplicate; first-seen ordering is preserved otherwise.
        """
        best_by_key: dict[tuple[str, str], ExtractedFact] = {}
        order: list[tuple[str, str]] = []
        for fact in facts:
            key = (fact.field_name, fact.value)
            existing = best_by_key.get(key)
            if existing is None:
                best_by_key[key] = fact
                order.append(key)
            elif fact.confidence > existing.confidence:
                best_by_key[key] = fact
        return [best_by_key[key] for key in order]

    def _compute_confidence(self, facts: list[ExtractedFact]) -> float:
        """
        Deterministic, field-weighted average confidence for a company.

        Each fact contributes confidence * field_weight (see
        _FIELD_WEIGHT). Weighting by field importance — rather than a
        flat average of every fact's confidence — means a company with
        strong core-field evidence (name/industry/location/size) scores
        higher than one with only a pile of low-importance contact/social
        facts, even if both have the same total fact count.
        """
        if not facts:
            return 0.0
        weighted_sum = 0.0
        total_weight = 0.0
        for f in facts:
            weight = self._FIELD_WEIGHT.get(f.field_name, self._DEFAULT_FIELD_WEIGHT)
            weighted_sum += f.confidence * weight
            total_weight += weight
        return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0

    @staticmethod
    def _crawled_content_snippet(crawl_result: CrawlResult, max_chars: int = 3000) -> str:
        """
        Plain text pulled from the actually-crawled pages, for the
        post-crawl re-validation "snippet" parameter below.

        Passing "" here (as the original post-crawl re-validation did)
        meant that check could only ever catch noise visible in a
        company's NAME. That misses self-descriptions that live in body
        copy instead of the title — confirmed real case: Startup Savant's
        own /about-us page states "Media Kit and Advertise page" and
        describes itself as a "resource guide" with an editorial team,
        exactly the kind of phrase utils/validator.py's media-publication
        and directory checks already look for — but none of that appears
        in its <title>, so title-only re-validation missed it. Reusing
        the same BeautifulSoup get_text() approach already used by
        crawlers/extractors/industry.py's _text_from_page, rather than
        introducing a second way to turn HTML into text.

        Deliberately concatenates ALL successfully crawled pages (not
        just the homepage) — the confirmed leak was specifically on
        /about-us, not the homepage, so limiting this to one page would
        have missed the exact case it exists to catch.
        """
        chunks: list[str] = []
        for page in crawl_result.successful_pages:
            try:
                soup = BeautifulSoup(page.html, "lxml")
                chunks.append(soup.get_text(" ", strip=True))
            except Exception:
                continue
        return " ".join(chunks)[:max_chars]

    @staticmethod
    def _best_field(facts: list[ExtractedFact], field: str) -> str:
        """
        Highest-confidence value for a field, or "" if none extracted.
        Shared by the post-crawl validation check (research_company) and
        _build_company's own `best()` closure — same selection rule,
        used in two places, so it lives once here instead of twice.
        """
        candidates = [f for f in facts if f.field_name == field]
        return max(candidates, key=lambda f: f.confidence).value if candidates else ""

    def _build_company(
        self,
        company_id: str, search_id: str, domain: str, website: str,
        discovery_source_url: str, discovery_snippet: str,
        facts: list[ExtractedFact], crawl_result: CrawlResult,
    ) -> Company:

        def best(field: str) -> str:
            return self._best_field(facts, field)

        def all_values(field: str) -> list[str]:
            seen: set[str] = set()
            result: list[str] = []
            for f in facts:
                if f.field_name == field and f.value not in seen:
                    seen.add(f.value)
                    result.append(f.value)
            return result

        def flag(field: str) -> bool:
            return any(f.field_name == field and f.value.lower() == "true" for f in facts)

        social: dict[str, str] = {}
        for field, key in [
            (F_SOCIAL_LINKEDIN, "linkedin"), (F_SOCIAL_TWITTER, "twitter"),
            (F_SOCIAL_FACEBOOK, "facebook"), (F_SOCIAL_INSTAGRAM, "instagram"),
        ]:
            val = best(field)
            if val:
                social[key] = val

        confidence = self._compute_confidence(facts)

        return Company(
            id=company_id,
            search_id=search_id,
            name=best(F_NAME) or domain,
            domain=domain,
            website=website,
            discovery_source_url=discovery_source_url,
            discovery_snippet=discovery_snippet,
            location=best(F_LOCATION),
            industry=best(F_INDUSTRY),
            employee_range=best(F_EMPLOYEE_RANGE),
            services=all_values(F_SERVICE),
            technologies=all_values(F_TECHNOLOGY),
            hiring_signals=flag(F_HIRING_SIGNAL),
            job_titles_found=all_values(F_JOB_TITLE),
            social_profiles=social,
            contact_pages=all_values(F_CONTACT_PAGE),
            contact_emails=all_values(F_CONTACT_EMAIL),
            contact_phones=all_values(F_CONTACT_PHONE),
            booking_detected=bool(best(F_BOOKING_WIDGET)),
            p1_pages_crawled=crawl_result.pages_crawled,
            p1_pages_failed=crawl_result.pages_failed,
            data_confidence=confidence,
            research_status=ResearchStatus.COMPLETE,
            researched_at=datetime.now(timezone.utc),
        )

    def _persist_company(self, company: Company, company_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._db.table("companies").upsert({
                "id": company_id,
                "search_id": company.search_id,
                "name": company.name,
                "domain": company.domain,
                "website": company.website,
                "discovery_source_url": company.discovery_source_url,
                "discovery_snippet": company.discovery_snippet,
                "location": company.location,
                "industry": company.industry,
                "employee_range": company.employee_range,
                "services": company.services,
                "technologies": company.technologies,
                "hiring_signals": company.hiring_signals,
                "job_titles_found": company.job_titles_found,
                "social_profiles": company.social_profiles,
                "contact_pages": company.contact_pages,
                "contact_emails": company.contact_emails,
                "contact_phones": company.contact_phones,
                "booking_detected": company.booking_detected,
                "p1_pages_crawled": company.p1_pages_crawled,
                "p1_pages_failed": company.p1_pages_failed,
                "data_confidence": company.data_confidence,
                "research_status": company.research_status,
                "researched_at": now,
                "created_at": now,
            }, on_conflict="search_id,domain").execute()
        except Exception as exc:
            logger.error("persist_company_failed", domain=company.domain, error=str(exc))
            raise

    def _persist_evidence(self, company_id: str, facts: list[ExtractedFact]) -> None:
        if not facts:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [{
            "id": str(uuid.uuid4()),
            "company_id": company_id,
            "source_url": f.source_url,
            "source_type": f.source_type,
            "trust_level": f.trust_level,
            "field_name": f.field_name,
            "value": f.value,
            "confidence": f.confidence,
            "extracted_at": now,
        } for f in facts]
        try:
            self._db.table("evidence").insert(rows).execute()
        except Exception as exc:
            logger.error("persist_evidence_failed", company_id=company_id, error=str(exc))

    def _promote_discovered(self, discovered_id: str, company_id: str) -> None:
        try:
            self._db.table("discovered_companies").update(
                {"promoted_to_company_id": company_id}
            ).eq("id", discovered_id).execute()
        except Exception as exc:
            logger.warning("promote_failed", discovered_id=discovered_id, error=str(exc))

    def _handle_failed(
        self, discovered_id: str, search_id: str, domain: str, website: str,
        discovery_source_url: str, discovery_snippet: str, crawl_result: CrawlResult,
    ) -> ResearchResponse:
        company_id = str(uuid.uuid4())
        company = Company(
            id=company_id, search_id=search_id, name=domain, domain=domain,
            website=website, discovery_source_url=discovery_source_url,
            discovery_snippet=discovery_snippet,
            p1_pages_crawled=crawl_result.pages_crawled,
            p1_pages_failed=crawl_result.pages_failed,
            data_confidence=0.0, research_status=ResearchStatus.FAILED,
            researched_at=datetime.now(timezone.utc),
        )
        try:
            self._persist_company(company, company_id)
            self._promote_discovered(discovered_id, company_id)
        except Exception:
            pass
        logger.warning("research_failed", domain=domain, error=crawl_result.error)
        return ResearchResponse(
            company=company, evidence_count=0,
            p1_pages_crawled=crawl_result.pages_crawled,
            p1_pages_failed=crawl_result.pages_failed,
            data_confidence=0.0,
        )

    def _handle_skipped(
        self, discovered_id: str, search_id: str, domain: str, website: str,
        discovery_source_url: str, discovery_snippet: str, reason: str,
    ) -> ResearchResponse:
        company_id = str(uuid.uuid4())
        company = Company(
            id=company_id, search_id=search_id, name=domain, domain=domain,
            website=website, discovery_source_url=discovery_source_url,
            discovery_snippet=discovery_snippet,
            data_confidence=0.0, research_status=ResearchStatus.SKIPPED,
            researched_at=datetime.now(timezone.utc),
        )
        try:
            self._persist_company(company, company_id)
            self._promote_discovered(discovered_id, company_id)
        except Exception:
            pass
        logger.info("research_skipped", domain=domain, reason=reason)
        return ResearchResponse(
            company=company, evidence_count=0,
            p1_pages_crawled=[], p1_pages_failed=[], data_confidence=0.0,
        )

    def _load_discovered(self, discovered_id: str) -> dict | None:
        try:
            result = (
                self._db.table("discovered_companies")
                .select("id, search_id, domain, website, source_url, snippet")
                .eq("id", discovered_id)
                .single()
                .execute()
            )
            return result.data
        except Exception as exc:
            logger.warning("load_discovered_failed", id=discovered_id, error=str(exc))
            return None

    def _load_unpromoted(self, search_id: str, limit: int) -> list[dict]:
        try:
            result = (
                self._db.table("discovered_companies")
                .select("id, search_id, domain, website, source_url, snippet")
                .eq("search_id", search_id)
                .is_("promoted_to_company_id", "null")
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as exc:
            logger.error("load_unpromoted_failed", search_id=search_id, error=str(exc))
            return []