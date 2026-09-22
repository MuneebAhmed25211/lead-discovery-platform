"""
utils/wikidata_check.py
────────────────────────
A structural upgrade to noise-filtering, discussed directly with the user
in response to "isn't there a better method than scraping websites and
guessing from prose?"

Every other check in utils/validator.py works the same way: find a real
leaked case (europa.eu, BetaKit, RETTC, ...), read its own self-description,
and hand-write a regex that catches that exact phrasing. That works, but
it's fundamentally reactive — it only ever catches a category AFTER it's
already leaked once, and it's blind to any phrasing variant the regex
author didn't happen to see.

Wikidata sidesteps that for one broad, high-value slice of the problem:
any entity notable enough to have its own Wikidata page (which covers
essentially every government body, national/regional news outlet, and
major nonprofit/NGO — exactly the categories that have been leaking) has
already been classified by Wikidata's community-maintained ontology via
the "instance of" (P31) property. Instead of inferring "is this a real
business" from scraped prose, this asks a structured, pre-vetted database
that already answered that question once, for everyone, rather than
re-deriving it from scratch per candidate. This is the same principle
mature commercial lead-gen tools (ZoomInfo, Clearbit, Apollo) rely
on — curated structured data instead of raw-text inference — implemented
here via a free, public, no-API-key-required source instead of a paid
firmographic vendor.

Deliberate limits, stated up front rather than discovered the hard way:
  - Most small private companies (the actual bulk of any realistic ICP)
    have NO Wikidata page at all. A miss here is the overwhelmingly
    common case and must never be treated as a rejection — only an
    exact, confident match against the disqualifying type list below
    should ever reject a candidate. Absence of evidence is not evidence
    of absence.
  - This is a genuinely different mechanism from utils/validator.py's
    regex checks, not a replacement for them — most noise (vendors,
    directories, personal blogs) will never have a Wikidata page and
    still needs the existing pattern-based checks.
  - Network calls to an external service can fail or time out in
    production the way any HTTP call can. A failure here degrades
    gracefully to "no signal" (treated identically to "no Wikidata
    page found") — it must never block or fail the research pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from core.logging import get_logger

logger = get_logger(__name__)

_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_REQUEST_TIMEOUT = 4.0  # seconds — this must never meaningfully slow down
                        # a research run; a slow/unreachable Wikidata is
                        # "no signal," not a blocker.

# Wikidata Q-IDs for entity types that disqualify a candidate as a lead,
# regardless of any ICP — none of these are ever the kind of "operating
# business that could realistically buy a B2B product/service" a lead
# discovery tool should be surfacing. Scoped deliberately narrow and
# unambiguous: this list trades recall for precision, since a wrong
# rejection here (a real company incorrectly sharing a name with, say, a
# defunct newspaper) is worse than a miss the existing regex checks might
# still catch from the crawled page itself.
#
# Each Q-ID resolved and confirmed via Wikidata's own item pages, not
# guessed from memory.
_DISQUALIFYING_TYPE_QIDS: dict[str, str] = {
    "Q327333": "government agency",
    "Q3624078": "sovereign state",
    "Q43229_intergovernmental": "intergovernmental organization",  # placeholder key, see note below
    "Q11032": "newspaper",
    "Q1110794": "online newspaper",
    "Q41298": "magazine",
    "Q1667921": "online magazine",
    "Q1193236": "news agency",
    "Q17155032": "advocacy group",
    "Q7278": "political party",
    "Q163740": "nonprofit organization",
    "Q79913": "non-governmental organization",
    "Q4671277": "academic institution",
    "Q1664720": "institute",
    "Q1530022": "religious organization",
    "Q245065": "intergovernmental organization",
    "Q28419196": "regional economic development organization",
    # Added after a confirmed live leak under a local-service-business ICP
    # (restaurant/food-industry discovery queries): MSN surfaced as a
    # "Food & Beverage" candidate. Its Wikidata "instance of" is literally
    # "web portal" (Q186165, aka internet portal / portal site) —
    # confirmed directly on Wikidata's own item page for MSN (Q192845),
    # not guessed. This is the same "content aggregator, not an operating
    # business" category as the existing directory/aggregator checks in
    # utils/validator.py, just for large general-news/content portals
    # specifically, which the domain keyword/exact-list approach there
    # can't fully enumerate any more than it can enumerate every trade
    # publication.
    "Q186165": "web portal",
    # Added same run: "95.7 KEZJ" surfaced as a "Food & Beverage" lead —
    # a commercial radio station, not caught by the existing
    # _fails_public_broadcasting_check in utils/validator.py (that check
    # is scoped to PBS/NPR-style PUBLIC broadcasting only, not commercial
    # stations). "broadcaster" (Q15265344) and "television station"
    # (Q1616075) confirmed directly on Wikidata's own item pages — both
    # describe organizations that transmit radio/TV programming, never a
    # realistic B2B/local-service lead under any ICP.
    # NOTE: a dedicated "radio station" QID (the more specific type an
    # individual station like KEZJ is more likely tagged with than the
    # broader "broadcaster" class) could not be verified via a live
    # Wikidata lookup in this session — search access was intermittently
    # failing. "broadcaster" is added now as the verified, safe parent
    # class; whether it alone catches KEZJ-style stations (vs. needing
    # the more specific "radio station" QID too) should be confirmed on
    # the next test run before treating radio-station leakage as closed.
    "Q15265344": "broadcaster",
    "Q1616075": "television station",
}

# The registry above has one placeholder key ("Q43229_intergovernmental")
# because Q43229 is actually "organization" (too broad/generic to use
# alone) — corrected here rather than left wrong: intergovernmental
# organization is properly Q245065, already listed above. Remove the
# placeholder at lookup time so a typo doesn't silently create a dead
# entry that can never match.
_DISQUALIFYING_TYPE_QIDS.pop("Q43229_intergovernmental", None)


@dataclass(frozen=True)
class WikidataCheckResult:
    checked: bool  # False if the check couldn't run (network error, no
                   # match, timeout) — callers must treat this the same
                   # as "passed", never as a rejection.
    flagged: bool
    matched_qid: str | None = None
    matched_type_label: str | None = None
    entity_label: str | None = None


_NO_SIGNAL = WikidataCheckResult(checked=False, flagged=False)


def _normalize_for_match(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


async def _search_entity(client: httpx.AsyncClient, name: str) -> tuple[str, str] | None:
    """
    Look up a company name against Wikidata's search-by-label endpoint.
    Returns (qid, matched_label) for the top hit ONLY if its label or an
    alias is a close match to the input name — a loose/fuzzy top result
    (Wikidata's search is label-substring-based, not exact) is worse than
    no result at all here, since a wrong match could reject a real
    company that merely shares a word with something else's name.
    """
    try:
        resp = await client.get(
            _WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": name,
                "language": "en",
                "format": "json",
                "limit": 5,
                "type": "item",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.info("wikidata_search_failed", name=name, error=str(exc))
        return None

    # Defensive: don't assume the parsed JSON has the exact shape
    # expected. `data.get("search", [])` only falls back to [] when the
    # "search" key is MISSING — if Wikidata ever returns it present but
    # explicitly null (e.g. a malformed or edge-case response), .get()
    # returns None instead of the default, and iterating that crashes.
    # Same defensive pattern applied to _fetch_instance_of_labels below.
    if not isinstance(data, dict):
        return None
    target = _normalize_for_match(name)
    for hit in data.get("search") or []:
        label = hit.get("label", "")
        aliases = hit.get("aliases", []) or []
        candidates = [label, *aliases]
        if any(_normalize_for_match(c) == target for c in candidates if c):
            return hit["id"], label
    return None


async def _fetch_instance_of_labels(client: httpx.AsyncClient, qid: str) -> dict[str, str]:
    """
    Fetch the P31 ("instance of") claims for an entity, returning a dict
    of {target_qid: english_label}. Two-step: get the target Q-IDs from
    the entity's own claims, then a second batched call to resolve those
    Q-IDs to human-readable English labels.
    """
    try:
        resp = await client.get(
            _WIKIDATA_API,
            params={
                "action": "wbgetclaims",
                "entity": qid,
                "property": "P31",
                "format": "json",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        claims_data = resp.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.info("wikidata_claims_failed", qid=qid, error=str(exc))
        return {}

    if not isinstance(claims_data, dict):
        return {}
    # Same defensive pattern as _search_entity above: each .get() here
    # could legitimately return None instead of falling through to its
    # default if the corresponding key is present but explicitly null in
    # Wikidata's response (e.g. a merged/redirected entity) — chaining
    # straight into another .get() on that would crash uncaught.
    target_qids: list[str] = []
    for claim in (claims_data.get("claims") or {}).get("P31") or []:
        try:
            target_qids.append(claim["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError, TypeError):
            continue

    if not target_qids:
        return {}

    try:
        resp = await client.get(
            _WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(target_qids),
                "props": "labels",
                "languages": "en",
                "format": "json",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        entities_data = resp.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.info("wikidata_labels_failed", qids=target_qids, error=str(exc))
        return {}

    if not isinstance(entities_data, dict):
        return {}
    labels: dict[str, str] = {}
    for target_qid, entity in (entities_data.get("entities") or {}).items():
        if not isinstance(entity, dict):
            continue
        label = (entity.get("labels") or {}).get("en") or {}
        label = label.get("value") if isinstance(label, dict) else None
        if label:
            labels[target_qid] = label
    return labels


async def check_wikidata_entity_type(
    client: httpx.AsyncClient | None, company_name: str
) -> WikidataCheckResult:
    """
    Look up a company name on Wikidata and flag it if its "instance of"
    property matches a known disqualifying entity type (government body,
    news outlet, nonprofit, etc.) — see module docstring for rationale
    and limits. Any failure (no match, network error, ambiguous name)
    returns "no signal," never a rejection: this check only ever adds
    confidence, it never removes it on its own absence.

    `client` is optional. Confirmed real bug: research_service.py's
    `research_company()` has always accepted `client: httpx.AsyncClient
    | None = None` — the single-company `/api/v1/research/company`
    endpoint calls it with no client at all (crawl_company() already
    handles that internally by making its own), and nothing else in the
    function touched `client` again after crawling. This function was
    the first thing to reuse that same variable post-crawl, assumed it
    was always a real client, and crashed every single research call on
    that endpoint with `'NoneType' object has no attribute 'get'` the
    moment it tried `client.get(...)`. Rather than push the burden of
    "remember to check for None" onto every caller (an easy mistake to
    repeat), this function is now self-sufficient: it opens its own
    short-lived client when none is supplied.
    """
    if not company_name or len(company_name) < 2:
        return _NO_SIGNAL

    if client is None:
        async with httpx.AsyncClient() as owned_client:
            return await _check_with_client(owned_client, company_name)
    return await _check_with_client(client, company_name)


async def _check_with_client(
    client: httpx.AsyncClient, company_name: str
) -> WikidataCheckResult:
    match = await _search_entity(client, company_name)
    if match is None:
        return _NO_SIGNAL
    qid, matched_label = match

    type_labels = await _fetch_instance_of_labels(client, qid)
    if not type_labels:
        return _NO_SIGNAL

    for target_qid, label in type_labels.items():
        if target_qid in _DISQUALIFYING_TYPE_QIDS:
            return WikidataCheckResult(
                checked=True,
                flagged=True,
                matched_qid=target_qid,
                matched_type_label=_DISQUALIFYING_TYPE_QIDS[target_qid],
                entity_label=matched_label,
            )

    return WikidataCheckResult(checked=True, flagged=False, entity_label=matched_label)