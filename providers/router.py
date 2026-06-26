"""
providers/router.py
────────────────────
SearchProviderRouter — orchestrates the Serper → Tavily fallback chain.

Fallback rules (exactly as specified):
  1. Serper raises (timeout/HTTP error/malformed response) → try Tavily
  2. Serper succeeds but returns zero results               → try Tavily
  3. Tavily also fails after Serper failed/empty             → return empty
     result for this query, marked as failed (not raised)

This router NEVER raises. Every possible outcome — success via Serper,
success via Tavily fallback, or total failure — is represented as a
ProviderSearchOutcome with succeeded=True/False. The discovery service
reads this flag; it never has to wrap router calls in try/except.

Caching:
  Cache key:  f"disc:{sha256(normalized_query)}"
  TTL:        settings.CACHE_DISCOVERY_TTL_HOURS (2 hours by default)
  Invalidation: TTL expiry only. Results are query-keyed, not company-keyed,
                so there is no manual invalidation trigger — a query either
                still means the same thing within 2 hours or it doesn't
                matter (company existence doesn't change in hours).
  What is cached: the NORMALIZED list[SearchResult], not raw provider JSON.
                  This means a cache hit skips both the API call AND the
                  parsing step — pure return of validated Pydantic objects
                  (re-validated from the cached dicts on read).
"""

from __future__ import annotations

import hashlib

from cache.supabase_cache import SupabaseCache
from core.config import settings
from core.logging import get_logger
from providers.base import SearchProvider, SearchProviderError
from providers.serper import SerperProvider
from providers.tavily import TavilyProvider
from schemas.discovery import ProviderSearchOutcome, SearchResult

logger = get_logger(__name__)


class SearchProviderRouter:
    """
    Routes a search query through Serper first, falling back to Tavily.

    Usage:
        router = SearchProviderRouter(cache=cache)
        outcome = await router.search("e-commerce companies hiring USA")
        if outcome.succeeded:
            for result in outcome.results:
                ...
    """

    def __init__(
        self,
        cache: SupabaseCache,
        serper: SearchProvider | None = None,
        tavily: SearchProvider | None = None,
    ) -> None:
        self._cache = cache
        self._serper = serper or SerperProvider()
        self._tavily = tavily or TavilyProvider()

    async def search(self, query: str, num_results: int = 10) -> ProviderSearchOutcome:
        """
        Execute one search query through the fallback chain.

        Returns a ProviderSearchOutcome describing exactly what happened —
        never raises. Callers should check `.succeeded` and `.results`
        rather than catching exceptions.
        """
        cache_key = self._cache_key(query)

        cached = await self._cache.get(cache_key)
        if cached is not None:
            results = [SearchResult(**r) for r in cached.get("results", [])]
            logger.info("discovery_cache_hit", query=query, count=len(results))
            return ProviderSearchOutcome(
                query=query,
                results=results,
                provider_used=cached.get("provider_used", "serper"),
                fallback_triggered=cached.get("fallback_triggered", False),
                cached=True,
                succeeded=True,
            )

        outcome = await self._search_live(query, num_results)

        if outcome.succeeded and outcome.results:
            await self._cache.set(
                cache_key,
                {
                    "results": [r.model_dump() for r in outcome.results],
                    "provider_used": outcome.provider_used,
                    "fallback_triggered": outcome.fallback_triggered,
                },
                ttl_hours=settings.CACHE_DISCOVERY_TTL_HOURS,
            )

        return outcome

    async def _search_live(self, query: str, num_results: int) -> ProviderSearchOutcome:
        """Run the actual Serper → Tavily fallback chain (no cache involved)."""

        serper_results, serper_error = await self._try_provider(self._serper, query, num_results)

        if serper_error is None and serper_results:
            logger.info("discovery_serper_success", query=query, count=len(serper_results))
            return ProviderSearchOutcome(
                query=query,
                results=serper_results,
                provider_used="serper",
                fallback_triggered=False,
                succeeded=True,
            )

        # Serper either raised or returned zero results — try Tavily.
        reason = serper_error or "zero results"
        logger.info("discovery_fallback_to_tavily", query=query, reason=reason)

        tavily_results, tavily_error = await self._try_provider(self._tavily, query, num_results)

        if tavily_error is None:
            # Tavily succeeded — even if it also returned zero results,
            # that is a legitimate empty answer, not a failure.
            logger.info("discovery_tavily_success", query=query, count=len(tavily_results))
            return ProviderSearchOutcome(
                query=query,
                results=tavily_results,
                provider_used="tavily",
                fallback_triggered=True,
                succeeded=True,
            )

        # Both providers failed.
        combined_error = f"serper: {reason}; tavily: {tavily_error}"
        logger.error("discovery_all_providers_failed", query=query, error=combined_error)
        return ProviderSearchOutcome(
            query=query,
            results=[],
            provider_used="none",
            fallback_triggered=True,
            succeeded=False,
            error=combined_error,
        )

    async def _try_provider(
        self, provider: SearchProvider, query: str, num_results: int
    ) -> tuple[list[SearchResult], str | None]:
        """
        Call a single provider, catching SearchProviderError.

        Returns (results, error_message). error_message is None on success
        (including a successful-but-empty result set).
        """
        try:
            results = await provider.search(query, num_results=num_results)
            return results, None
        except SearchProviderError as exc:
            logger.warning("provider_call_failed", provider=provider.name, query=query, error=exc.message)
            return [], exc.message
        except Exception as exc:
            # Defensive catch-all: a provider implementation bug should
            # never escape the router and crash the discovery endpoint.
            logger.error(
                "provider_unexpected_error",
                provider=provider.name,
                query=query,
                error=str(exc),
                exc_info=True,
            )
            return [], f"unexpected error: {exc}"

    @staticmethod
    def _cache_key(query: str) -> str:
        normalized = " ".join(query.lower().split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"disc:{digest}"
