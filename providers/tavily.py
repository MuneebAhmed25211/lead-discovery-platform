"""
providers/tavily.py
────────────────────
Tavily search provider — fallback discovery source.

Tavily is purpose-built for AI agent consumption and returns cleaner,
pre-processed results than a raw SERP scrape. It is used only as a
fallback when Serper fails or returns zero usable candidates, since it
has a smaller free tier and costs more per query at scale.

API docs: https://docs.tavily.com/docs/rest-api/api-reference

Timeout and retry behavior:
  - Single attempt per call, same timeout budget as Serper.
  - NO internal retry — if Tavily also fails, the router returns a
    graceful empty result for that query rather than retrying further.
"""

from __future__ import annotations

from typing import Any

import httpx

from core.config import settings
from core.logging import get_logger
from providers.base import SearchProvider, SearchProviderError
from schemas.discovery import SearchResult

logger = get_logger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


class TavilyProvider(SearchProvider):

    name = "tavily"

    def __init__(self, api_key: str | None = None, timeout: int | None = None) -> None:
        self._api_key = api_key or settings.TAVILY_API_KEY
        self._timeout = timeout or settings.CRAWLER_TIMEOUT_SECONDS

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            raise SearchProviderError(self.name, "TAVILY_API_KEY is not configured")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": min(num_results, 20),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(_TAVILY_URL, json=payload)
        except httpx.TimeoutException as exc:
            logger.warning("tavily_timeout", query=query, timeout=self._timeout)
            raise SearchProviderError(self.name, f"Request timed out after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            logger.warning("tavily_http_error", query=query, error=str(exc))
            raise SearchProviderError(self.name, f"HTTP error: {exc}") from exc

        if response.status_code == 401 or response.status_code == 403:
            raise SearchProviderError(self.name, f"Authentication failed (HTTP {response.status_code})")
        if response.status_code == 429:
            raise SearchProviderError(self.name, "Rate limit exceeded")
        if response.status_code != 200:
            raise SearchProviderError(self.name, f"Unexpected status code {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise SearchProviderError(self.name, f"Malformed JSON response: {exc}") from exc

        return self._parse_results(body, query)

    def _parse_results(self, body: dict[str, Any], query: str) -> list[SearchResult]:
        """
        Extract results from Tavily's response shape.

        Tavily returns a top-level 'results' array, each entry having
        'title', 'url', and 'content' (used here as the snippet).

        As with Serper, malformed individual entries are skipped rather
        than failing the entire query.
        """
        items = body.get("results", [])
        if not isinstance(items, list):
            logger.warning("tavily_unexpected_shape", query=query, body_keys=list(body.keys()))
            return []

        results: list[SearchResult] = []
        for entry in items:
            try:
                title = entry.get("title", "")
                url = entry.get("url", "")
                snippet = entry.get("content", "")
                if not title or not url:
                    continue
                results.append(
                    SearchResult(title=title, url=url, snippet=snippet, provider="tavily")
                )
            except (ValueError, TypeError) as exc:
                logger.debug("tavily_entry_skipped", query=query, error=str(exc))
                continue

        return results
