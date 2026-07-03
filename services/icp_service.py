"""
services/icp_service.py
───────────────────────
ICP (Ideal Customer Profile) generation service.

This is Groq call #1 in the pipeline.

Responsibilities:
  1. Check cache — return immediately if description was seen before
  2. Build the Groq prompt with strict JSON output instructions
  3. Call Groq via LangChain (with retry decorator)
  4. Parse and validate the response with Pydantic
  5. Store the ICP record in Supabase
  6. Cache the result for 24 hours
  7. Return the structured ICP

Prompt design principles:
  - System prompt defines the JSON schema strictly
  - User prompt contains only the service description
  - Temperature is low (0.3) — we want consistent, structured output
  - We ask for exactly 3–5 search queries — no more (API cost control)
  - No hallucinated company names or URLs allowed in this output
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from supabase import Client

from cache.supabase_cache import SupabaseCache
from core.config import settings
from core.errors import GroqError
from core.logging import get_logger
from schemas.icp import ICP, ICPGenerateResponse, ICPRecord
from utils.retry import groq_retry

logger = get_logger(__name__)

# Strict JSON schema the model must follow
_ICP_SYSTEM_PROMPT = """You are an expert B2B sales strategist. Your job is to analyse a service description and produce a structured Ideal Customer Profile (ICP) for the BUYERS of that service — not for businesses offering a similar service.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no preamble.

The JSON must follow this exact schema:
{
  "target_industries": ["string"],
  "target_company_sizes": ["string"],
  "target_locations": ["string"],
  "pain_points": ["string"],
  "buying_triggers": ["string"],
  "positive_signals": ["string"],
  "disqualifying_signals": ["string"],
  "search_queries": ["string"],
  "summary": "string"
}

CRITICAL DISTINCTION — read this before anything else:
The service_description describes WHAT THE USER SELLS. Your job is to identify WHO BUYS IT — i.e. industries that have a business need for this service but do not provide it themselves.

NEVER target the same industry as the service being sold, and never target close competitors or companies likely to build the capability in-house.

Examples of this distinction:
- Service: "Full Stack AI Development" / "AI automation systems"
  WRONG targets: AI companies, software development agencies, SaaS companies, tech startups (these build AI themselves or are competitors)
  RIGHT targets: healthcare providers, manufacturing companies, logistics firms, law firms, accounting firms, retailers, e-commerce businesses, hospitality groups, construction companies, educational institutions, insurance agencies — industries that NEED automation but do not build software as their core business
- Service: "AI chatbots for dental clinics"
  RIGHT target: dental clinics and dental groups (the service description already names the buyer industry explicitly)
- Service: "Marketing agency for SaaS companies"
  RIGHT target: SaaS and software companies (again, explicitly named as the buyer)
- Service: "Bookkeeping and tax services for small businesses"
  RIGHT targets: small businesses across various non-accounting industries — retail shops, restaurants, contractors, salons, clinics

Rule of thumb: if the service_description explicitly names who it serves (e.g. "for dental clinics", "for SaaS companies"), target that named industry directly. If the service_description only describes a capability without naming a buyer industry (e.g. "AI development", "automation systems", "web development"), you must infer realistic buyer industries — businesses that would hire an outside specialist for this capability rather than build it internally. Default to a diverse spread of non-technology industries in that case.

Rules:
- target_industries: 2–5 specific BUYER industries (not the seller's own industry). e.g. for "AI automation systems": "healthcare providers", "logistics companies", "manufacturing firms", "law firms" — NOT "AI companies" or "software companies"
- target_company_sizes: use these exact ranges only: "1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"
- target_locations: 2–4 countries or regions the user can realistically serve
- pain_points: 3–5 specific operational problems this service solves FOR THE BUYER (not problems the seller has)
- buying_triggers: 3–5 events that make a buyer company likely to purchase now (e.g. "hiring an Operations Manager", "opened a second location", "manual process complaints in job postings")
- positive_signals: 3–5 things to look for on a BUYER's website that indicate strong fit (e.g. signs of manual processes, no existing automation, growth signals)
- disqualifying_signals: 1–3 signals that indicate poor fit (e.g. "already has an in-house engineering team", "is itself a software/AI company", "government-funded")
- search_queries: EXACTLY 4–5 Google search queries designed to find REAL BUYER BUSINESS WEBSITES in the target industries — never queries that would surface companies offering the same service as the user. Each query must:
  * Target commercial businesses in a BUYER industry, not the seller's own industry
  * Use operator-style terms that surface company homepages (e.g. "careers", "contact us", "our team")
  * Include industry + business-intent keywords specific to a buyer vertical (e.g. "manufacturing company hiring operations manager", "law firm multiple offices careers", "logistics company expansion new warehouse")
  * Avoid generic terms that return directories (e.g. avoid "best AI companies", "top automation agencies", "list of")
  * Vary the buyer industry across the 4–5 queries when target_industries has multiple distinct verticals, so discovery isn't concentrated in just one industry
  * Example good queries for "Full Stack AI Development" (capability-only description, no named buyer):
    - manufacturing company hiring operations manager California
    - logistics company multiple warehouses careers Texas
    - law firm multiple offices expansion New York
    - healthcare clinic group patient intake manual process
    - retail chain inventory management hiring
  * Example good queries for "AI chatbots for dental clinics" (buyer named explicitly):
    - multi location dental group California careers
    - orthodontic practice management software contact
    - dental chain patient portal online booking California
    - dental office hiring front desk staff Texas
    - private dental group expansion new location
  * Do NOT include quotes inside queries
- summary: One paragraph (3–4 sentences) describing the ideal BUYER customer in plain English, explicitly noting their industry and why they would need this service rather than build it themselves.

Do not invent company names. Do not fabricate statistics. Only output the JSON object."""


class ICPService:

    def __init__(self, db: Client, cache: SupabaseCache) -> None:
        self._db = db
        self._cache = cache
        self._llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )

    async def generate(
        self,
        service_description: str,
        search_id: str,
    ) -> ICPGenerateResponse:
        """
        Generate or retrieve a cached ICP for the given service description.
        """
        description_hash = _hash_description(service_description)
        cache_key = SupabaseCache.make_key("icp", description_hash)

        # ── Cache hit ─────────────────────────────────────────────────────────
        cached = await self._cache.get(cache_key)
        if cached:
            logger.info("icp_cache_hit", hash=description_hash[:8])
            icp = ICP(**cached["icp"])
            return ICPGenerateResponse(
                icp_id=cached["icp_id"],
                search_id=search_id,
                icp=icp,
                cached=True,
            )

        # ── Groq call #1 ──────────────────────────────────────────────────────
        logger.info("icp_groq_call", hash=description_hash[:8])
        raw_json = await self._call_groq(service_description)
        icp = self._parse_icp(raw_json)

        # ── Persist to Supabase ───────────────────────────────────────────────
        icp_id = str(uuid.uuid4())
        await self._store(
            icp_id=icp_id,
            search_id=search_id,
            description_hash=description_hash,
            icp=icp,
            raw_json=raw_json,
        )

        # ── Cache result ──────────────────────────────────────────────────────
        await self._cache.set(
            cache_key,
            {"icp_id": icp_id, "icp": icp.model_dump()},
            ttl_hours=settings.CACHE_ICP_TTL_HOURS,
        )

        return ICPGenerateResponse(
            icp_id=icp_id,
            search_id=search_id,
            icp=icp,
            cached=False,
        )

    @groq_retry
    async def _call_groq(self, service_description: str) -> dict[str, Any]:
        """
        Call Groq and return the parsed JSON response.

        The @groq_retry decorator handles retries and rate limit detection.
        """
        messages = [
            SystemMessage(content=_ICP_SYSTEM_PROMPT),
            HumanMessage(content=f"Service description: {service_description}"),
        ]

        response = await self._llm.ainvoke(messages)
        content = response.content.strip()

        # Strip markdown code fences if the model added them despite instructions
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("icp_json_parse_failed", content=content[:200], error=str(exc))
            raise GroqError(
                operation="icp_generation",
                cause=f"Model returned invalid JSON: {str(exc)}",
            )

    def _parse_icp(self, raw: dict[str, Any]) -> ICP:
        """
        Validate and coerce raw Groq output into a typed ICP object.

        Pydantic v2 validation will raise ValidationError if the model
        returned a structurally invalid response. We let this propagate
        so the retry decorator can catch it.
        """
        # Sanitise: ensure all list fields are actually lists
        for field in [
            "target_industries", "target_company_sizes", "target_locations",
            "pain_points", "buying_triggers", "positive_signals",
            "disqualifying_signals", "search_queries",
        ]:
            if field in raw and isinstance(raw[field], str):
                raw[field] = [raw[field]]

        # Clamp search queries to 5 max, ensure at least 4 if possible
        if "search_queries" in raw and len(raw["search_queries"]) > 5:
            raw["search_queries"] = raw["search_queries"][:5]

        return ICP(**raw)

    async def _store(
        self,
        icp_id: str,
        search_id: str,
        description_hash: str,
        icp: ICP,
        raw_json: dict,
    ) -> None:
        try:
            self._db.table("icps").insert({
                "id": icp_id,
                "search_id": search_id,
                "description_hash": description_hash,
                "target_industries": icp.target_industries,
                "target_company_sizes": icp.target_company_sizes,
                "target_locations": icp.target_locations,
                "pain_points": icp.pain_points,
                "buying_triggers": icp.buying_triggers,
                "positive_signals": icp.positive_signals,
                "disqualifying_signals": icp.disqualifying_signals,
                "search_queries": icp.search_queries,
                "summary": icp.summary,
                "raw_json": raw_json,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as exc:
            logger.warning("icp_store_failed", icp_id=icp_id, error=str(exc))


def _hash_description(description: str) -> str:
    """Deterministic 16-char hash of a normalised description string."""
    normalised = " ".join(description.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]
