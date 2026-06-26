"""
providers/serper.py
────────────────────
Serper.dev search provider — primary discovery source.

Serper proxies Google Search results via a simple POST API. It is the
primary provider because it is cheapest at volume and returns results
closest to what a human would find Googling the same query.

API docs: https://serper.dev/playground

Timeout and retry behavior:
  - Single attempt per call, 8-second timeout (CRAWLER_TIMEOUT_SECONDS reused
    from existing config — discovery and crawling share the same timeout
    budget philosophy).
  - NO internal retry — a failed Serper call falls forward to Tavily via the
    router rather than retrying Serper itself. See providers/router.py for
    the fallback decision.
"""

from __future__ import annotations

from typing import Any

import httpx

from core.config import settings
from core.logging import get_logger
from providers.base import SearchProvider, SearchProviderError
from schemas.discovery import SearchResult

logger = get_logger(__name__)

_SERPER_URL = "https://google.serper.dev/search"


class SerperProvider(SearchProvider):

    name = "serper"

    def __init__(self, api_key: str | None = None, timeout: int | None = None) -> None:
        self._api_key = api_key or settings.SERPER_API_KEY
        self._timeout = timeout or settings.CRAWLER_TIMEOUT_SECONDS

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            raise SearchProviderError(self.name, "SERPER_API_KEY is not configured")

        payload = {"q": query, "num": min(num_results, 20)}
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(_SERPER_URL, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning("serper_timeout", query=query, timeout=self._timeout)
            raise SearchProviderError(self.name, f"Request timed out after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            logger.warning("serper_http_error", query=query, error=str(exc))
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
        Extract organic results from Serper's response shape.

        Serper's response includes 'organic' (the actual search results we
        want), plus 'knowledgeGraph', 'peopleAlsoAsk', 'relatedSearches' —
        none of which are usable company candidates, so they're ignored.

        Malformed individual entries are skipped (logged at debug level)
        rather than failing the entire query — one bad entry should never
        discard otherwise-valid results.
        """
        organic = body.get("organic", [])
        if not isinstance(organic, list):
            logger.warning("serper_unexpected_shape", query=query, body_keys=list(body.keys()))
            return []

        results: list[SearchResult] = []
        for entry in organic:
            try:
                title = entry.get("title", "")
                link = entry.get("link", "")
                snippet = entry.get("snippet", "")
                if not title or not link:
                    continue
                results.append(
                    SearchResult(title=title, url=link, snippet=snippet, provider="serper")
                )
            except (ValueError, TypeError) as exc:
                logger.debug("serper_entry_skipped", query=query, error=str(exc))
                continue

        return results
