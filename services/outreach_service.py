"""
services/outreach_service.py
─────────────────────────────
Outreach service — Groq call #3 (final Groq call) in the pipeline.

Responsibilities:
  1. Check cache — return immediately if outreach was already generated
     for this company
  2. Load the deterministic Company record, its Evidence rows, and the
     existing Qualification (Groq call #2's output) for that company
  3. Build a strict, evidence-and-qualification-grounded Groq prompt —
     the model is only given facts already extracted/derived by Phase 4
     and Phase 5; it may not invent anything, it may only WRITE
  4. Call Groq via LangChain (with retry decorator, same as ICP/qualify)
  5. Parse and validate the response with Pydantic
  6. Store the outreach record in Supabase (`outreach_messages` table)
  7. Cache the result for CACHE_OUTREACH_TTL_HOURS
  8. Return the structured Outreach

Evidence preservation: this service is READ-ONLY against `companies`,
`evidence`, and `qualifications` — it never writes to any of them. The
only table this service writes to is `outreach_messages`.

Prompt design principles (mirrors services/qualify_service.py):
  - System prompt defines the JSON schema strictly
  - User prompt contains ONLY facts already on record — deterministic
    Company fields, a compact evidence listing, and the existing
    Qualification (pain_points, suggested_services, outreach_angle,
    buying_triggers, why_qualified) — never open-ended web knowledge
  - Temperature is low (settings.GROQ_TEMPERATURE) for consistent output
  - Explicit "no hallucination, no business logic" instruction: Groq's
    only job here is to WRITE the messages, not decide fit or invent
    facts — that reasoning already happened in qualification (Groq #2)
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from supabase import Client

from cache.supabase_cache import SupabaseCache
from core.config import settings
from core.errors import GroqError, NotFoundError
from core.logging import get_logger
from schemas.qualification import Outreach, OutreachVariant
from utils.retry import groq_retry

logger = get_logger(__name__)

_OUTREACH_SYSTEM_PROMPT = """You are an expert B2B sales copywriter. Your ONLY job is to WRITE four short outreach messages using facts that have already been verified and reasoned about — you do not decide fit, invent facts, or perform any new analysis.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no preamble.

The JSON must follow this exact schema:
{
  "linkedin_message": {"body": "string"},
  "cold_email": {"subject": "string", "body": "string"},
  "audit_opener": {"body": "string"},
  "followup_message": {"body": "string"}
}

CRITICAL RULE — grounding only:
You will be given a deterministic Company record, a list of Evidence facts (each with the exact page it was found on), and an existing Qualification analysis (why_qualified, pain_points, suggested_services, outreach_angle, buying_triggers — already produced by a separate reasoning step). You may ONLY reference facts and conclusions that appear in what you are given. Do not invent company details, statistics, names of people, revenue figures, technologies, contact channels, or events that are not explicitly present in the material given to you. Do not perform new fit analysis or contradict the qualification you were given — your job is purely to phrase what is already known into outreach copy.

CRITICAL RULE — empty means absent:
If a field is an empty list ([]) or marked "(not detected)", that thing was NOT found — never phrase it as if it exists. For example, if contact_phones is [], never write "you can be reached by phone" or "only lists email and phone" — only reference the contact channels that are actually populated.

CRITICAL RULE — copy identifiers exactly:
Any email address, phone number, URL, or domain you reference must be copied character-for-character from the material given — never paraphrase, autocomplete, or lightly reword an identifier. If you are not fully certain of the exact string, omit it and refer to the channel generically instead (e.g. "your listed email" instead of guessing at the address).

Field instructions:
- linkedin_message.body: A short LinkedIn connection note or first DM (400-500 characters max). Opens with a specific, real observation about the company (drawn from evidence or the qualification), not a generic greeting. No "I hope this finds you well", no "I came across your profile", no corporate buzzwords ("synergy", "leverage", "circle back", "reach out to discuss synergies"). Sounds like one person messaging another. Ends with a soft, low-pressure question or nudge — not a hard pitch.
- cold_email.subject: Under 60 characters, specific to this company (not generic like "Quick question" unless genuinely earned by context), no clickbait, no ALL CAPS, no excessive punctuation.
- cold_email.body: 3-5 short paragraphs. First line/paragraph names something specific and real about the company (from evidence or qualification) — never opens with "I hope you're doing well" or "My name is X and I work at Y". Names the pain point(s) and the suggested_service(s) it addresses. Ends with a single, easy, low-friction ask (e.g. "Worth a 15-minute call?" or "Want me to send over what that would look like?") — not a hard close.
- audit_opener.body: A short, no-pressure offer to do a free/light audit or teardown of one specific thing detected in the evidence (e.g. their booking flow, their hiring page, their site's automation gaps) — framed as genuinely useful before any ask for payment or a call. 2-4 sentences.
- followup_message.body: A brief, low-pressure follow-up assuming the first message got no reply — references the original angle briefly without repeating it verbatim, adds one new specific detail or angle if the evidence supports it, and makes it easy to say no. 2-4 sentences.

Tone rules (apply to all four):
- Human-sounding: write like a real freelancer/consultant messaging a prospect, not like an AI or a template.
- No generic filler openers, no "I hope this email finds you well", no "As a [industry] company, you understand...".
- No fabricated urgency, no fake scarcity, no guarantees of results.
- Never claim a call, meeting, or prior interaction that hasn't happened.
- Do not use the word "synergy" or similar corporate jargon.

Do not fabricate. Only output the JSON object."""


class OutreachService:

    def __init__(self, db: Client, cache: SupabaseCache) -> None:
        self._db = db
        self._cache = cache
        self._llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )

    async def generate_outreach(self, company_id: str) -> tuple[Outreach, bool]:
        """
        Generate personalized outreach for a single (already-qualified) company.

        Returns (Outreach, cached) — mirrors QualifyService.qualify_company's
        cache-hit/cache-miss contract.
        """
        cache_key = SupabaseCache.make_key("outreach", company_id)

        # ── Cache hit ─────────────────────────────────────────────────────────
        cached = await self._cache.get(cache_key)
        if cached:
            logger.info("outreach_cache_hit", company_id=company_id)
            return Outreach(**cached["outreach"]), True

        # ── Load deterministic inputs (Company, Evidence, Qualification) ────────
        company = self._load_company(company_id)
        evidence = self._load_evidence(company_id)
        qualification = self._load_qualification(company_id)

        # ── Groq call #3 ─────────────────────────────────────────────────────
        logger.info("outreach_groq_call", company_id=company_id)
        raw_json = await self._call_groq(company, evidence, qualification)
        outreach = self._parse_outreach(raw_json, company_id)

        # ── Persist to Supabase (outreach_messages table ONLY) ─────────────────
        self._store(outreach)

        # ── Cache result ─────────────────────────────────────────────────────
        await self._cache.set(
            cache_key,
            {"outreach": outreach.model_dump(mode="json")},
            ttl_hours=settings.CACHE_OUTREACH_TTL_HOURS,
        )

        return outreach, False

    @groq_retry
    async def _call_groq(
        self, company: dict, evidence: list[dict], qualification: dict,
    ) -> dict[str, Any]:
        """
        Call Groq and return the parsed JSON response.

        The @groq_retry decorator handles retries and rate limit detection,
        identical to QualifyService._call_groq.
        """
        user_prompt = self._build_user_prompt(company, evidence, qualification)
        messages = [
            SystemMessage(content=_OUTREACH_SYSTEM_PROMPT),
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
            logger.error("outreach_json_parse_failed", content=content[:200], error=str(exc))
            raise GroqError(
                operation="outreach",
                cause=f"Model returned invalid JSON: {str(exc)}",
            )

    def _build_user_prompt(
        self, company: dict, evidence: list[dict], qualification: dict,
    ) -> str:
        """
        Build the grounding context Groq is allowed to write from.

        Only deterministic, already-extracted facts and the existing
        qualification analysis go in here — nothing is looked up live and
        no new fit reasoning is requested from the model.
        """
        evidence_lines = "\n".join(
            f"  - [{e['field_name']}] {e['value']} (source: {e.get('source_url', 'unknown')})"
            for e in evidence[:40]   # cap prompt size — same cap as qualify_service
        ) or "  (no additional evidence rows)"

        return f"""Company record (deterministic, from Phase 4 research):
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
  contact_emails: {company.get('contact_emails', [])}
  contact_phones: {company.get('contact_phones', [])}
  contact_pages: {company.get('contact_pages', [])}

NOTE ON EMPTY FIELDS: an empty list ([]) means that channel or fact was
NOT found — it does not exist as far as the evidence shows. Never imply
a contact_phones number exists if contact_phones is []. Same for any
other empty list or "(not detected)" field above.

Evidence (each fact traced to the exact page it came from):
{evidence_lines}

Existing qualification analysis (already reasoned — do not contradict, only write from it):
  qualification_status: {qualification.get('qualification_status')}
  why_qualified: {qualification.get('why_qualified')}
  pain_points: {qualification.get('pain_points', [])}
  suggested_services: {qualification.get('suggested_services', [])}
  outreach_angle: {qualification.get('outreach_angle')}
  buying_triggers: {qualification.get('buying_triggers', [])}
  evidence_summary: {qualification.get('evidence_summary')}

Write the four outreach messages now, grounded only in the material above. Output the JSON object now."""

    def _parse_outreach(self, raw: dict[str, Any], company_id: str) -> Outreach:
        """
        Validate and coerce raw Groq output into a typed Outreach.

        The `type` field on each OutreachVariant is set deterministically
        here (never by the model) since it's a fixed structural label, not
        content the model should be free to phrase.
        """
        try:
            linkedin = raw["linkedin_message"]
            cold_email = raw["cold_email"]
            audit_opener = raw["audit_opener"]
            followup = raw["followup_message"]
        except KeyError as exc:
            raise GroqError(
                operation="outreach",
                cause=f"Model response missing required field: {exc}",
            )

        outreach = Outreach(
            company_id=company_id,
            linkedin_message=OutreachVariant(
                type="linkedin", body=linkedin.get("body", "").strip(),
            ),
            cold_email=OutreachVariant(
                type="cold_email",
                subject=cold_email.get("subject", "").strip(),
                body=cold_email.get("body", "").strip(),
            ),
            audit_opener=OutreachVariant(
                type="audit_opener", body=audit_opener.get("body", "").strip(),
            ),
            followup_message=OutreachVariant(
                type="followup", body=followup.get("body", "").strip(),
            ),
            groq_model=settings.GROQ_MODEL,
        )
        return outreach

    def _store(self, o: Outreach) -> None:
        """Write ONLY to the outreach_messages table — never companies/evidence/qualifications."""
        try:
            self._db.table("outreach_messages").upsert({
                "id": str(uuid.uuid4()),
                "company_id": o.company_id,
                "linkedin_message": o.linkedin_message.body,
                "cold_email_subject": o.cold_email.subject,
                "cold_email_body": o.cold_email.body,
                "audit_opener": o.audit_opener.body,
                "followup_message": o.followup_message.body,
                "groq_model": o.groq_model,
                "created_at": o.created_at.isoformat(),
            }, on_conflict="company_id").execute()
        except Exception as exc:
            logger.warning("outreach_store_failed", company_id=o.company_id, error=str(exc))

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

    def _load_qualification(self, company_id: str) -> dict:
        """
        Outreach is written from the existing Qualification (Groq #2's
        output), not recomputed from the ICP directly — this keeps Groq #3
        a pure writing step with no new business-logic reasoning, per the
        Phase 6 architectural rule (Groq only performs writing).
        """
        result = (
            self._db.table("qualifications")
            .select("qualification_status, why_qualified, pain_points, "
                    "suggested_services, outreach_angle, buying_triggers, evidence_summary")
            .eq("company_id", company_id).single().execute()
        )
        if not result.data:
            raise NotFoundError("Qualification", company_id)
        return result.data