"""
services/qualify_service.py
────────────────────────────
Qualification service — Groq call #2 in the pipeline.

Responsibilities:
  1. Check cache — return immediately if this (company, icp) pair was
     qualified before
  2. Load the deterministic Company record, its Evidence rows, the ICP,
     and (if it exists) its LeadScore breakdown
  3. Build a strict, evidence-grounded Groq prompt — the model is only
     given facts already extracted by Phase 4; it may not invent anything
  4. Call Groq via LangChain (with retry decorator, same as ICP generation)
  5. Parse and validate the response with Pydantic
  6. Store the qualification record in Supabase (`qualifications` table)
  7. Cache the result for CACHE_QUALIFY_TTL_HOURS
  8. Return the structured Qualification

Evidence preservation: this service is READ-ONLY against `companies` and
`evidence` — it never writes to either table. Research remains the sole
source of truth for what a company IS; qualification only interprets
that existing evidence to decide fit. The only table this service writes
to is `qualifications`.

Prompt design principles (mirrors services/icp_service.py):
  - System prompt defines the JSON schema strictly
  - User prompt contains ONLY facts already on record — deterministic
    Company fields, a compact evidence listing, and the ICP — never
    open-ended web knowledge
  - Temperature is low (settings.GROQ_TEMPERATURE) for consistent output
  - Explicit "no hallucination" instruction: every claim must trace to a
    fact given in the prompt
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from supabase import Client

from cache.supabase_cache import SupabaseCache
from core.config import settings
from core.errors import DatabaseError, GroqError, NotFoundError
from core.logging import get_logger
from schemas.qualification import Qualification, QualifyBatchResponse
from utils.retry import groq_retry

logger = get_logger(__name__)

_BATCH_CONCURRENCY = 3   # same cap used by ResearchService.research_batch

_QUALIFY_SYSTEM_PROMPT = """You are an expert B2B sales analyst. Your job is to decide whether a prospect company is a good fit for the service the user sells, using ONLY the facts provided to you below.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no preamble.

The JSON must follow this exact schema:
{
  "qualification_status": "qualified" | "partially_qualified" | "not_qualified",
  "confidence": 0.0-1.0,
  "why_qualified": "string",
  "pain_points": ["string"],
  "suggested_services": ["string"],
  "outreach_angle": "string",
  "buying_triggers": ["string"],
  "evidence_summary": "string"
}

CRITICAL RULE — grounding only:
You will be given a deterministic Company record, a list of Evidence facts (each with the exact page it was found on), and the user's Ideal Customer Profile (ICP). You may ONLY reason from these facts. Do not invent company details, statistics, employee counts, revenue figures, technologies, or events that are not explicitly present in the evidence given to you. If the evidence is thin, say so honestly in why_qualified and lower confidence accordingly — do not compensate for missing evidence by inventing plausible-sounding facts.

Field instructions:
- qualification_status: "qualified" if the company clearly matches the ICP's target_industries/target_company_sizes and shows positive_signals or buying_triggers with no disqualifying_signals present; "not_qualified" if a disqualifying_signal is clearly present OR the company clearly falls outside the ICP's target industries/sizes; "partially_qualified" for everything in between (decent fit but missing key signals, or fit is unclear from the evidence given).
- ENTITY-TYPE CHECK — apply this before the industry/size check above, and always "not_qualified" (regardless of confidence) if it fails: is this actually an operating commercial business that could realistically buy the user's service? "not_qualified" if the evidence describes a government body or public institution, a news outlet/magazine/blog/content site, a directory/listing/aggregator site, a trade association or nonprofit, an academic/research institution, or a company that is itself a vendor/competitor selling the same category of service the user sells. This check exists because deterministic keyword filtering upstream (utils/validator.py) cannot anticipate every phrasing of these categories — a hobby blog about birds, a government portal, an industry directory — but you can recognize the shape of the entity from its own evidence (self-description, page structure, "about us" language) even when no single keyword rule matches it. Judge this from what the evidence actually says the company IS and does, not from its industry label alone — a company mislabeled "Healthcare" by the deterministic industry extractor is still not a real prospect if its own evidence describes it as a personal advice blog or content hub, and a company labeled "Technology" is still not a real prospect if the evidence describes it as a national government's official web portal.
- confidence: how much evidence actually supports your verdict — a company with rich, specific evidence (multiple matching pain points, technologies, hiring signals) should get high confidence; a company with sparse or ambiguous evidence should get low confidence, REGARDLESS of which qualification_status you chose. Confidence measures evidence strength, not fit quality.
- why_qualified: 2-4 sentences citing SPECIFIC facts from the evidence (e.g. "no online booking system detected", "actively hiring for Front Desk roles") — never generic claims.
- pain_points: 1-5 concrete operational pain points this specific company likely has, inferred from its own evidence (not generic industry pain points). Before concluding evidence is too thin, actively work with what deterministic signals ARE present — hiring for specific roles, technologies in use, presence/absence of a booking widget, service descriptions, employee range — each of these can support a specific, evidence-cited pain point even when there's no single explicit statement of pain. Only fall back to noting thin evidence if truly none of these signals are present, or if the company is disqualified (in which case there is no real pain-point case to construct).
  Worked examples of deriving a pain point from a single deterministic signal (do this kind of inference, don't just wait for an explicit statement of pain):
    - hiring for "Product Manager" + "Head of Product" + no automation/AI tooling in technologies -> "scaling product output without engineering bandwidth to build AI-assisted features in-house"
    - booking_detected: false + industry is a service business (dental, healthcare, real estate, etc.) -> "manual, phone/email-based scheduling with no online booking system, adding administrative overhead"
    - employee_range 201-500 + hiring_signals true + no mobile app mentioned in services -> "growing mid-size company without a mobile presence, a customer-experience gap at this scale"
    - technologies list shows a CRM (HubSpot/Salesforce) but nothing else automation-related -> "manual, non-automated workflows around the CRM despite investment in it"
- suggested_services: 1-3 services from the user's own offering (given below) that map directly to the pain points you identified. Apply the same rule as above — derive from whatever deterministic signals exist before concluding there's nothing to suggest.
- outreach_angle: one sentence — the single most compelling, evidence-grounded reason to reach out now.
- buying_triggers: which of the ICP's buying_triggers (if any) are actually evidenced for this company — empty list if none apply.
- evidence_summary: one sentence naming which pages/fields the analysis relied on (e.g. "Based on the About and Careers pages, and detected technologies.").

Do not fabricate. Only output the JSON object."""


class QualifyService:

    def __init__(self, db: Client, cache: SupabaseCache) -> None:
        self._db = db
        self._cache = cache
        self._llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )

    async def qualify_company(
        self,
        company_id: str,
        icp_id: str,
    ) -> tuple[Qualification, bool]:
        """
        Qualify a single company against an ICP.

        Returns (Qualification, cached) — mirrors ICPService.generate's
        cache-hit/cache-miss contract.
        """
        cache_key = SupabaseCache.make_key("qualify", company_id, icp_id)

        # ── Cache hit ─────────────────────────────────────────────────────────
        cached = await self._cache.get(cache_key)
        if cached:
            logger.info("qualify_cache_hit", company_id=company_id, icp_id=icp_id)
            return Qualification(**cached["qualification"]), True

        # ── Load deterministic inputs (Company, Evidence, ICP) ──────────────────
        company = self._load_company(company_id)
        evidence = self._load_evidence(company_id)
        icp = self._load_icp(icp_id)
        lead_score = self._load_lead_score(company_id)   # optional, may be None
        service_description = self._load_service_description(icp.get("search_id", ""))

        # ── Groq call #2 ──────────────────────────────────────────────────────
        logger.info("qualify_groq_call", company_id=company_id, icp_id=icp_id)
        raw_json = await self._call_groq(company, evidence, icp, lead_score, service_description)
        qualification = self._parse_qualification(raw_json, company_id)

        # ── Persist to Supabase (qualifications table ONLY — never companies/evidence) ─
        self._store(qualification)

        # ── Cache result ──────────────────────────────────────────────────────
        await self._cache.set(
            cache_key,
            {"qualification": qualification.model_dump(mode="json")},
            ttl_hours=settings.CACHE_QUALIFY_TTL_HOURS,
        )

        return qualification, False

    async def qualify_batch(
        self,
        search_id: str,
        icp_id: str,
        min_score: int = 40,  # see schemas/qualification.py QualifyBatchRequest.min_score for the full rationale
    ) -> QualifyBatchResponse:
        """
        Qualify every researched company for a search whose lead_score
        total_score >= min_score. Bounded concurrency, same pattern as
        ResearchService.research_batch and ScoringService.score_batch.
        """
        candidates = self._load_scored_companies(search_id, min_score)
        qualifications: list[Qualification] = []
        skipped = 0
        semaphore = asyncio.Semaphore(_BATCH_CONCURRENCY)

        async def qualify_one(company_id: str) -> None:
            nonlocal skipped
            async with semaphore:
                try:
                    qual, _cached = await self.qualify_company(
                        company_id=company_id, icp_id=icp_id,
                    )
                    qualifications.append(qual)
                except Exception as exc:
                    logger.warning("batch_qualify_failed", company_id=company_id, error=str(exc))
                    skipped += 1

        await asyncio.gather(*[qualify_one(cid) for cid in candidates])

        return QualifyBatchResponse(
            qualified=len(qualifications), skipped=skipped, total=len(candidates),
            qualifications=qualifications,
        )

    @groq_retry
    async def _call_groq(
        self, company: dict, evidence: list[dict], icp: dict,
        lead_score: dict | None, service_description: str,
    ) -> dict[str, Any]:
        """
        Call Groq and return the parsed JSON response.

        The @groq_retry decorator handles retries and rate limit detection,
        identical to ICPService._call_groq.
        """
        user_prompt = self._build_user_prompt(company, evidence, icp, lead_score, service_description)
        messages = [
            SystemMessage(content=_QUALIFY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await self._llm.ainvoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("qualify_json_parse_failed", content=content[:200], error=str(exc))
            raise GroqError(
                operation="qualification",
                cause=f"Model returned invalid JSON: {str(exc)}",
            )

    def _build_user_prompt(
        self, company: dict, evidence: list[dict], icp: dict,
        lead_score: dict | None, service_description: str,
    ) -> str:
        """
        Build the grounding context Groq is allowed to reason from.

        Only deterministic, already-extracted facts go in here — nothing
        is looked up live and nothing is summarized from general web
        knowledge about the company.
        """
        evidence_lines = "\n".join(
            f"  - [{e['field_name']}] {e['value']} (source: {e.get('source_url', 'unknown')}, "
            f"confidence: {e.get('confidence', 0):.2f})"
            for e in evidence[:40]   # cap prompt size — most companies have well under 40 facts
        ) or "  (no additional evidence rows)"

        score_block = ""
        if lead_score:
            score_block = (
                f"\nDeterministic lead score (already computed, for context only — "
                f"do not just restate it): {lead_score.get('total_score')}/100 "
                f"({lead_score.get('score_category')})\n"
            )
            breakdown = lead_score.get("score_breakdown") or {}
            # Surface the *reasons* behind each dimension (not just the
            # number) — these are the deterministic engine's own findings
            # on automation opportunity, growth signals, hiring, and any
            # disqualifier match, and are useful grounding for identifying
            # buying signals / automation opportunities / pain points below.
            dim_lines = [
                f"  - {dim}: {info.get('reason')}"
                for dim, info in breakdown.items()
                if isinstance(info, dict) and info.get("reason")
            ]
            if dim_lines:
                score_block += (
                    "Deterministic scoring reasons per dimension "
                    "(already-extracted facts, not open-ended reasoning):\n"
                    + "\n".join(dim_lines) + "\n"
                )

        return f"""User's service offering: {service_description or "(not provided)"}

ICP (Ideal Customer Profile):
  target_industries: {icp.get('target_industries', [])}
  target_company_sizes: {icp.get('target_company_sizes', [])}
  target_locations: {icp.get('target_locations', [])}
  pain_points: {icp.get('pain_points', [])}
  buying_triggers: {icp.get('buying_triggers', [])}
  positive_signals: {icp.get('positive_signals', [])}
  disqualifying_signals: {icp.get('disqualifying_signals', [])}

Company record (deterministic, from Phase 4 research):
  name: {company.get('name')}
  domain: {company.get('domain')}
  industry: {company.get('industry') or '(not detected)'}
  location: {company.get('location') or '(not detected)'}
  employee_range: {company.get('employee_range') or '(not detected)'}
  services: {company.get('services', [])}
  technologies: {company.get('technologies', [])}
  hiring_signals: {company.get('hiring_signals', False)}
  job_titles_found: {company.get('job_titles_found', [])}
  booking_detected: {company.get('booking_detected', False)}
  data_confidence: {company.get('data_confidence', 0)}
{score_block}
Evidence (each fact traced to the exact page it came from):
{evidence_lines}

Analyse this company against the ICP using ONLY the facts above. Output the JSON object now."""

    def _parse_qualification(self, raw: dict[str, Any], company_id: str) -> Qualification:
        """
        Validate and coerce raw Groq output into a typed Qualification.
        Pydantic v2 validation raises on structurally invalid output,
        which the retry decorator's caller can surface as a GroqError.
        """
        for field in ["pain_points", "suggested_services", "buying_triggers"]:
            if field in raw and isinstance(raw[field], str):
                raw[field] = [raw[field]]

        # _QUALIFY_SYSTEM_PROMPT explicitly tells Groq it's fine — preferred,
        # even — to say evidence is thin rather than invent plausible-sounding
        # facts. For a low-evidence company that produces a legitimate empty
        # pain_points/suggested_services list. But the schema requires >=1
        # item on both (Phase 6 outreach generation and CSV export expect
        # non-empty lists), so an honest empty response was crashing this
        # endpoint with a raw Pydantic validation error instead of returning
        # a valid, low-confidence qualification. Backfill an explicit
        # "insufficient evidence" placeholder rather than loosening the
        # schema, so the honest signal survives without breaking that
        # non-empty contract downstream.
        if not raw.get("pain_points"):
            raw["pain_points"] = ["Insufficient evidence to identify specific pain points"]
        if not raw.get("suggested_services"):
            raw["suggested_services"] = [
                "Insufficient evidence — recommend further discovery before pitching specific services"
            ]

        raw["company_id"] = company_id
        raw["groq_model"] = settings.GROQ_MODEL
        return Qualification(**raw)

    def _store(self, q: Qualification) -> None:
        """
        Write ONLY to the qualifications table — never companies or
        evidence.

        BUG FIX: this used to catch every exception and only log a
        warning, letting qualify_company() return success (and cache the
        result) even when the row was never actually persisted. The
        person would then see a fully-qualified company in the UI — real
        "why it qualifies" reasoning, pain points, angle, all correctly
        computed by Groq — but the moment they clicked "Generate outreach
        messages", outreach_service._load_qualification() does a fresh DB
        read for this same company_id, finds zero rows, and crashes with
        a raw "PGRST116 / Cannot coerce the result to a single JSON
        object" error that surfaces as "Failed to fetch" with no
        indication of what actually went wrong. Confirmed real case: this
        exact sequence happened live for "Warner & Warner". Raising here
        means a failed qualification write surfaces immediately (as a
        skipped/failed candidate in qualify_batch's response) instead of
        silently degrading into a working-looking qualification that
        can never generate outreach.
        """
        try:
            self._db.table("qualifications").upsert({
                "id": str(uuid.uuid4()),
                "company_id": q.company_id,
                "qualification_status": q.qualification_status,
                "confidence": q.confidence,
                "why_qualified": q.why_qualified,
                "pain_points": q.pain_points,
                "suggested_services": q.suggested_services,
                "outreach_angle": q.outreach_angle,
                "buying_triggers": q.buying_triggers,
                "evidence_summary": q.evidence_summary,
                "groq_model": q.groq_model,
                "created_at": q.created_at.isoformat(),
            }, on_conflict="company_id").execute()
        except Exception as exc:
            logger.error("qualification_store_failed", company_id=q.company_id, error=str(exc))
            raise DatabaseError(operation="upsert", table="qualifications", cause=str(exc)) from exc

    # ── Loaders (read-only) ──────────────────────────────────────────────────

    def _load_company(self, company_id: str) -> dict:
        result = self._db.table("companies").select("*").eq("id", company_id).single().execute()
        if not result.data:
            raise NotFoundError("Company", company_id)
        return result.data

    def _load_evidence(self, company_id: str) -> list[dict]:
        result = (
            self._db.table("evidence")
            .select("id, field_name, value, confidence, source_url, trust_level")
            .eq("company_id", company_id).lte("trust_level", 2).execute()
        )
        return result.data or []

    def _load_icp(self, icp_id: str) -> dict:
        result = (
            self._db.table("icps")
            .select("search_id, target_industries, target_company_sizes, target_locations, "
                    "pain_points, buying_triggers, positive_signals, disqualifying_signals")
            .eq("id", icp_id).single().execute()
        )
        if not result.data:
            raise NotFoundError("ICP", icp_id)
        return result.data

    def _load_service_description(self, search_id: str) -> str:
        """
        The user's service offering isn't part of QualifyRequest (company_id +
        icp_id only) — it's looked up via icp.search_id -> searches.service_description
        so the API contract doesn't need a new required field.
        """
        if not search_id:
            return ""
        try:
            result = (
                self._db.table("searches")
                .select("service_description")
                .eq("id", search_id).single().execute()
            )
            return (result.data or {}).get("service_description", "")
        except Exception:
            return ""

    def _load_lead_score(self, company_id: str) -> dict | None:
        """
        Optional context — qualification can run before or without a score.

        Includes score_breakdown (already computed and stored by
        ScoringService, no extra computation here) so the qualification
        prompt can ground its buying-signal / automation-opportunity /
        pain-point analysis in the same deterministic reasoning the scorer
        already produced, rather than re-deriving it from raw evidence
        alone.
        """
        try:
            result = (
                self._db.table("lead_scores")
                .select("total_score, score_category, score_breakdown")
                .eq("company_id", company_id).single().execute()
            )
            return result.data or None
        except Exception:
            return None

    def _load_scored_companies(self, search_id: str, min_score: int) -> list[str]:
        """
        Company IDs for a search whose lead_score total_score >= min_score.
        Mirrors ScoringService's own per-company-check pattern rather than
        a SQL join, since the Supabase client's fluent API doesn't support
        cross-table joins directly.
        """
        companies = (
            self._db.table("companies")
            .select("id")
            .eq("search_id", search_id).eq("research_status", "complete").execute()
        )
        company_ids = [c["id"] for c in (companies.data or [])]
        if not company_ids:
            return []

        scores = (
            self._db.table("lead_scores")
            .select("company_id, total_score")
            .in_("company_id", company_ids).execute()
        )
        return [
            s["company_id"] for s in (scores.data or [])
            if s.get("total_score", 0) >= min_score
        ]