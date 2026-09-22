"""
crawlers/extractors/industry.py
────────────────────────────────
Classify a company's industry from crawled page content.

Approach: keyword taxonomy matching against homepage and about page text.
This is entirely deterministic — no LLM.

Industry taxonomy is deliberately broad and industry-agnostic so the
system works for any niche: dental, SaaS, law, accounting, consulting,
manufacturing, healthcare, etc.

Priority:
  1. schema.org @type maps to industry             (0.90)
  2. Keyword taxonomy match on homepage/about text  (0.70–0.85)
  3. Meta description keyword match                 (0.65)

Returns at most one industry fact — the highest-confidence match.
If multiple taxonomy categories match, the one with the most keyword
hits wins.
"""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    ExtractedFact,
    F_INDUSTRY,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult


# ── Schema.org type → industry string ─────────────────────────────────────────

_SCHEMA_TYPE_MAP: dict[str, str] = {
    "DentalClinic": "Dental",
    "Dentist": "Dental",
    "MedicalBusiness": "Healthcare",
    "MedicalClinic": "Healthcare",
    "Hospital": "Healthcare",
    "Physician": "Healthcare",
    "LegalService": "Legal",
    "Attorney": "Legal",
    "Lawyer": "Legal",
    "AccountingService": "Accounting & Finance",
    "FinancialService": "Accounting & Finance",
    # Added: confirmed gap — CIBC (a bank) surfaced under an ICP that
    # explicitly listed "financial services firms" as a target industry,
    # but had nowhere correct to land: schema.org's "FinancialService"
    # type (mapped above) is the generic parent used by advisors and
    # accountants, and there was no entry at all for its more specific
    # "BankOrCreditUnion" subtype — so banks fell through to the
    # keyword-based fallback and landed in "Retail" instead.
    "BankOrCreditUnion": "Banking & Financial Services",
    "RealEstateAgent": "Real Estate",
    "RealEstateBusiness": "Real Estate",
    "Restaurant": "Food & Beverage",
    "FoodEstablishment": "Food & Beverage",
    "Hotel": "Hospitality",
    "LodgingBusiness": "Hospitality",
    "AutoDealer": "Automotive",
    "AutoRepair": "Automotive",
    "HomeAndConstructionBusiness": "Construction & Home Services",
    "Plumber": "Construction & Home Services",
    "Electrician": "Construction & Home Services",
    "GeneralContractor": "Construction & Home Services",
    "BeautySalon": "Beauty & Wellness",
    "HealthAndBeautyBusiness": "Beauty & Wellness",
    "SportsActivityLocation": "Fitness & Sports",
    "ExerciseGym": "Fitness & Sports",
    "EducationalOrganization": "Education",
    "School": "Education",
    "CollegeOrUniversity": "Education",
    "InsuranceAgency": "Insurance",
    "TravelAgency": "Travel & Tourism",
    "ITCompany": "Technology",
    "SoftwareApplication": "Technology",
    "ConsultingBusiness": "Consulting",
    "MarketingAgency": "Marketing & Advertising",
    "RecruitmentAgency": "Staffing & Recruitment",
    "MovingCompany": "Logistics & Moving",
    "Store": "Retail",
    "ClothingStore": "Retail",
    "ElectronicsStore": "Retail",
}


# ── Keyword taxonomy ───────────────────────────────────────────────────────────

# Maps industry label → list of keywords that suggest that industry.
# Keywords are matched against lowercased page text.
# More specific terms score higher in tiebreaking (longer = more specific).
_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "Dental": [
        "dental", "dentist", "orthodontic", "orthodontist", "oral health",
        "teeth whitening", "dental implants", "braces", "invisalign",
        "periodontal", "endodontic", "dental hygienist", "dental clinic",
        "dental practice", "dental group", "oral surgery",
    ],
    "Healthcare": [
        "medical", "clinic", "physician", "doctor", "patient care",
        "healthcare", "health care", "hospital", "urgent care",
        "physical therapy", "chiropractic", "optometry", "ophthalmology",
        "dermatology", "pediatric", "cardiology", "radiology",
        "nursing", "telehealth", "primary care",
    ],
    "Legal": [
        "law firm", "attorney", "lawyer", "legal services", "litigation",
        "esquire", "llp", "legal advice", "paralegal",
        "immigration", "criminal defense", "personal injury", "corporate law",
        # NOTE: "trademark", "patent", "intellectual property", and bare
        # "counsel" were removed — confirmed false-positive source. These
        # appear in ordinary corporate boilerplate on almost ANY company's
        # site regardless of industry ("e.l.f.(R) and all related
        # trademarks are owned by...", "protected by copyright and
        # trademark law", a "General Counsel" exec bio) — not specific to
        # being a law firm. Real example: e.l.f. Beauty (a cosmetics
        # company) was misclassified as "Legal" purely from its /about
        # page's standard trademark/copyright footer text.
    ],
    "Accounting & Finance": [
        "accounting", "accountant", "cpa", "bookkeeping", "tax preparation",
        "audit", "financial planning", "wealth management", "investment",
        "payroll", "controller", "cfo", "financial advisor", "tax filing",
        "budgeting", "forensic accounting",
    ],
    # Added: confirmed gap — see the BankOrCreditUnion schema.org mapping
    # above for the real case (CIBC misclassified as "Retail"). Banks
    # weren't just missing a schema mapping, they had no keyword-fallback
    # category either, so sites without schema.org markup fell through
    # to Retail's generic "product" keyword (banks constantly say
    # "banking products", "financial products") with nothing more
    # specific to outrank it. Kept distinct from "Accounting & Finance"
    # (advisor/bookkeeper-focused terms) since a bank is a different real
    # buyer profile than an accounting firm.
    "Banking & Financial Services": [
        "checking account", "savings account", "credit union",
        "commercial banking", "retail banking", "business banking",
        "mortgage lending", "loan officer", "branch locations",
        "deposit accounts", "credit card services", "atm locations",
        "bank branch", "chartered bank",
    ],
    "Technology": [
        "software", "saas", "app development", "mobile app", "web development",
        "cloud", "api", "platform", "startup", "tech company",
        "artificial intelligence", "machine learning", "cybersecurity",
        "data analytics", "devops", "it services", "it consulting",
        "enterprise software", "erp", "crm",
    ],
    "Marketing & Advertising": [
        "marketing agency", "digital marketing", "seo", "ppc", "content marketing",
        "social media marketing", "email marketing", "branding", "advertising",
        "pr agency", "public relations", "growth hacking", "lead generation",
        "inbound marketing", "copywriting", "media buying",
    ],
    "Consulting": [
        "consulting", "consultant", "advisory", "strategy", "management consulting",
        "business consulting", "process improvement", "operations consulting",
        "transformation", "change management",
    ],
    "Real Estate": [
        "real estate", "realtor", "property management", "commercial real estate",
        "residential real estate", "mortgage", "brokerage", "leasing",
        "property investment", "homebuyer", "reit",
    ],
    "Construction & Home Services": [
        "construction", "contractor", "general contractor", "roofing", "plumbing",
        "electrical", "hvac", "renovation", "remodeling", "landscaping",
        "flooring", "painting", "home improvement",
        # NOTE: "building" removed — confirmed false-positive source.
        # "building" is a common gerund verb in generic corporate mission
        # statements ("committed to building a more inclusive industry",
        # "building the future of X") that has nothing to do with
        # construction. Real construction companies are already reliably
        # caught by the more specific terms above.
    ],
    "Retail": [
        "e-commerce", "ecommerce", "retail", "online store", "shop",
        "merchandise", "product", "shopify", "woocommerce", "inventory",
        "brick and mortar", "point of sale", "consumer goods",
    ],
    "Logistics & Supply Chain": [
        "logistics", "freight", "shipping", "warehousing", "supply chain",
        "fulfillment", "distribution", "transportation", "trucking",
        "last mile", "3pl", "fleet management",
    ],
    "Education": [
        "school", "tutoring", "online course", "curriculum", "students",
        "learning management", "edtech", "training program", "certification",
        "e-learning", "lms", "higher education", "k-12",
    ],
    "Staffing & Recruitment": [
        "staffing agency", "recruitment", "headhunting", "talent acquisition",
        "hiring", "job placement", "temporary staffing", "executive search",
        "workforce", "hr services",
    ],
    "Food & Beverage": [
        "restaurant", "catering", "food service", "bakery", "cafe",
        "coffee shop", "brewery", "winery", "food delivery", "meal prep",
        "food manufacturer",
    ],
    "Fitness & Wellness": [
        "gym", "fitness", "yoga", "pilates", "personal trainer",
        "wellness", "spa", "massage", "nutrition", "health coach",
        "crossfit", "martial arts",
    ],
    "Manufacturing": [
        "manufacturing", "factory", "production", "fabrication", "machining",
        "assembly", "industrial", "original equipment manufacturer",
        "iso certified", "quality control", "lean manufacturing", "cnc",
    ],
    "Insurance": [
        "insurance", "insurer", "underwriting",
        "life insurance", "health insurance", "property insurance",
        "risk management", "reinsurance", "insurance agency",
        "insurance broker", "insurance quotes",
    ],
    # BUG FIX: this category was entirely missing from the keyword taxonomy
    # even though _SCHEMA_TYPE_MAP already maps AutoDealer/AutoRepair
    # schema.org types to "Automotive" — meaning any auto repair shop, body
    # shop, or dealership page WITHOUT schema.org markup (the vast majority
    # of small real-world sites) had no keyword path to this label at all
    # and instead won whichever unrelated category happened to rack up the
    # most incidental hits (e.g. "shop"/"store" -> Retail, "oem"/
    # "industrial" -> Manufacturing). Confirmed real case: testing the ICP
    # "work-order and estimate management software for auto repair shops"
    # produced 20 discovered companies and NOT ONE was classified
    # "Automotive" via keywords — Subaru of America came back "Retail",
    # Crash Champions (a collision-repair chain) came back "Manufacturing",
    # BizBen came back "Real Estate" — which then tanked industry_fit
    # (worth 25/100 points) for nearly every genuine prospect in the batch.
    "Automotive": [
        "auto repair", "auto body", "collision repair", "collision center",
        "car repair", "vehicle repair", "auto repair shop", "body shop",
        "mechanic shop", "auto service center", "auto service",
        "oil change", "brake repair", "transmission repair", "engine repair",
        "tire shop", "tire service", "wheel alignment", "muffler shop",
        "auto glass repair", "paintless dent repair", "auto dealership",
        "car dealership", "used car dealer", "new car dealer",
        "auto diagnostic", "vehicle diagnostic", "ase certified",
        "auto maintenance", "vehicle maintenance", "fleet maintenance",
    ],
    # BUG FIX: same missing-category pattern as Automotive above. Hotel/
    # LodgingBusiness schema.org types already map to "Hospitality" in
    # _SCHEMA_TYPE_MAP, but there was no keyword fallback for lodging
    # businesses without schema markup. Deliberately excludes cafe/
    # restaurant/bakery terms — those correctly belong to the existing
    # "Food & Beverage" category above; this one is scoped to lodging.
    # Confirmed real cases: "Hotel Boutique" (hotelboutiqueatgrandcentral.com)
    # and "Life Hotel New York" both came back "Insurance" instead of
    # Hospitality when tested against a boutique-hotel ICP.
    "Hospitality": [
        "boutique hotel", "bed and breakfast", "bed & breakfast",
        "guesthouse", "guest house", "hotel", "inn", "lodge", "resort",
        "hospitality group", "vacation rental", "room service",
        "concierge", "front desk", "check-in desk",
        # Hotel MANAGEMENT companies (operate hotels on behalf of owners)
        # describe themselves more abstractly than an individual hotel
        # does, and don't always say "hotel" prominently. Confirmed real
        # case: Coury Hospitality classified "Insurance" instead.
        "hospitality management", "hotel management company",
        "hotel management", "hotel operator", "manages hotels",
        "third-party hotel management",
    ],
    "Events & Weddings": [
        "wedding planner", "wedding planning", "wedding coordinator",
        "bridal", "event planner", "event planning", "event production",
        "event management company", "special events", "corporate events",
        "event coordination", "day-of coordination", "full-service event",
        "event agency", "experiential marketing agency", "event design",
    ],
}


# Word-boundary-safe compiled patterns for the taxonomy above.
#
# BUG FIX: naive substring containment (`kw in text`) causes false
# positives whenever a keyword is a substring of an unrelated, longer
# word — the concrete case that surfaced this: "hospital" (a Healthcare
# keyword) is a substring of "hospitality", so any hospitality/restaurant
# company whose site says "International Hospitality Management" or
# "hospitality professionals" was picking up a false Healthcare hit,
# which then beat or tied the correct "Food & Beverage" match (only
# "restaurant" hits it) — Healthcare being defined earlier in this dict
# meant it also won any tie. \b...\b boundaries fix this generally, not
# just for this one collision. An optional trailing "s" is allowed so
# regular plurals ("restaurant" -> "restaurants", "clinic" -> "clinics")
# still match — only the boundary *inside* a longer unrelated word
# (like "hospital"+"ity") is rejected.
_INDUSTRY_PATTERNS: dict[str, list[re.Pattern]] = {
    industry: [re.compile(r"\b" + re.escape(kw) + r"s?\b", re.IGNORECASE) for kw in keywords]
    for industry, keywords in _INDUSTRY_KEYWORDS.items()
}


def _text_from_page(soup: BeautifulSoup, max_chars: int = 5000) -> str:
    """Extract readable text from a page, capped to avoid regex slowdowns."""
    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "header"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:max_chars].lower()


# Every one of the "Technology" category's keywords ("software", "platform",
# "api", "cloud", "artificial intelligence"...) is a word a company that
# merely USES AI/software would say just as often as one that SELLS AI/
# software development services. Because _from_keywords picks whichever
# category racks up the most raw keyword hits, an actual dev/AI agency's
# page usually still gets classified as generic "Technology" — its
# service-specific phrases are a handful of hits against Technology's much
# longer, broader keyword list. This category is checked as its own pass
# (like schema.org) with a higher confidence floor, so a company that
# clearly markets itself with these specific phrases is classified
# correctly instead of losing a hit-count contest it shouldn't have to
# enter. This distinction also feeds industry_fit (a real buyer's industry
# no longer gets diluted into the same bucket as its own competitors) and
# lets scoring's vendor-disqualifier check trust the industry field
# directly (see _VENDOR_DISQUALIFIER_TRIGGERS / industry_suggests_vendor in
# services/scoring_service.py).
_AGENCY_INDICATOR_PHRASES = [
    "software development agency", "software development company",
    "custom software development", "software outsourcing",
    "outsourced development", "staff augmentation", "nearshore development",
    "offshore development", "dedicated development team",
    "digital agency", "web development agency", "web design agency",
    "app development agency", "app development company",
    "mobile app development company", "it consulting company",
    "software consultancy", "development shop", "ai consulting",
    "ai development agency", "ai development company",
    "generative ai consulting", "llm development", "chatbot development",
    "computer vision development", "managed service provider",
]
_AGENCY_INDICATOR_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(p) + r"s?\b", re.IGNORECASE) for p in _AGENCY_INDICATOR_PHRASES
]


def _from_agency_phrases(text: str, source_url: str) -> ExtractedFact | None:
    source_type = source_type_for_url(source_url)
    hits = sum(1 for p in _AGENCY_INDICATOR_PATTERNS if p.search(text))
    if hits == 0:
        return None
    # A single occurrence of e.g. "custom software development" or "ai
    # consulting" is unambiguous, decisive evidence — unlike a generic
    # Technology keyword ("software", "cloud", "platform") which any AI
    # buyer's site could just as easily mention. The floor is set above
    # Technology's own confidence ceiling (0.88) so a single hit here
    # reliably wins the cross-page "best fact" comparison in
    # extract_industry, rather than needing to out-count Technology's much
    # broader keyword list on the same page.
    confidence = min(0.90 + (hits - 1) * 0.01, 0.97)
    return ExtractedFact(
        field_name=F_INDUSTRY,
        value="AI/Software Development Agency",
        source_url=source_url,
        source_type=source_type,
        confidence=confidence,
    )


_ENTITY_TYPE_PATTERNS = {
    "Banking & Financial Services": [
        re.compile(r"\b(federal\s+)?credit\s+union\b", re.IGNORECASE),
        re.compile(r"\bsavings\s+bank\b", re.IGNORECASE),
        re.compile(r"\bnational\s+bank\b", re.IGNORECASE),
        re.compile(r"\bbank\s*&\s*trust\b", re.IGNORECASE),
    ],
}


def _from_entity_type_name(soup: BeautifulSoup, source_url: str) -> ExtractedFact | None:
    """
    Classify from the company's own name (title tag / og:site_name) when
    it unambiguously states its entity type — same architectural pattern
    as _from_agency_phrases: a decisive signal that shouldn't have to win
    a raw keyword hit-count contest against the rest of the page.

    Confirmed real gap: four different credit unions (A+ Federal Credit
    Union, Cy-Fair Federal Credit Union, First Basin Credit Union, Alive
    Credit Union) were misclassified as Education, Construction & Home
    Services, or Healthcare — not because the "Banking & Financial
    Services" keyword category was missing (it wasn't, and worked
    correctly for several other credit unions in the same batch), but
    because credit unions' own inner pages are dense with vocabulary
    tied to what they lend against or who they serve — student loan
    pages ("school", "education", "student"), mortgage pages ("home",
    "construction", "renovation"), health-savings-account pages
    ("health", "wellness") — which won the raw hit-count contest
    _from_keywords runs, since it has no way to know those hits describe
    a loan product rather than the company's own business. The
    company's literal name ("X Federal Credit Union") sidesteps that
    entirely: it's unambiguous regardless of which inner page happened
    to get crawled or how loan-product-heavy that page's content is.
    """
    site_name = ""
    og_tag = soup.find("meta", property="og:site_name")
    if og_tag and og_tag.get("content"):
        site_name = og_tag["content"]
    elif soup.title and soup.title.string:
        site_name = soup.title.string

    if not site_name:
        return None

    for industry, patterns in _ENTITY_TYPE_PATTERNS.items():
        if any(p.search(site_name) for p in patterns):
            # Above schema.org's fixed 0.90 and _from_keywords' 0.88
            # ceiling — same rationale as _from_agency_phrases' floor.
            return ExtractedFact(
                field_name=F_INDUSTRY,
                value=industry,
                source_url=source_url,
                source_type=source_type_for_url(source_url),
                confidence=0.93,
            )
    return None


_MARKET_RESEARCH_INDICATOR_PHRASES = [
    "market research firm", "market intelligence firm",
    "market research company", "market research and advisory",
    "syndicated research", "syndicated market research",
    "custom market research", "industry research reports",
    "market intelligence platform", "market intelligence reports",
    "research and advisory services",
]
_MARKET_RESEARCH_INDICATOR_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(p) + r"s?\b", re.IGNORECASE)
    for p in _MARKET_RESEARCH_INDICATOR_PHRASES
]


def _from_market_research_phrases(text: str, source_url: str) -> ExtractedFact | None:
    """
    Same architectural pattern as _from_agency_phrases, for the same
    reason: a market-research/analyst firm's site is saturated with
    generic vocabulary belonging to whatever industries its reports
    cover ("retail", "product", "inventory", "consumer goods" as report
    topics; "healthcare", "manufacturing" on another page) — none of
    which describes what the company itself does. That vocabulary wins
    _from_keywords' raw hit-count contest against almost any target
    category, so a decisive self-description phrase needs to short-
    circuit it the same way the agency-phrase pass already does.
    Confirmed real case: Mordor Intelligence classified "Retail" instead
    of the closer-fit "Consulting" bucket, surfaced as the top-scoring
    (68/100) result under a retail/e-commerce AI-agent ICP — the single
    most consequential instance of this failure mode seen so far, since
    it topped the list rather than sitting in the noise at the bottom.
    """
    source_type = source_type_for_url(source_url)
    hits = sum(1 for p in _MARKET_RESEARCH_INDICATOR_PATTERNS if p.search(text))
    if hits == 0:
        return None
    confidence = min(0.90 + (hits - 1) * 0.01, 0.97)
    return ExtractedFact(
        field_name=F_INDUSTRY,
        value="Consulting",
        source_url=source_url,
        source_type=source_type,
        confidence=confidence,
    )


def _from_schema_org(soup: BeautifulSoup, source_url: str) -> ExtractedFact | None:
    source_type = source_type_for_url(source_url)
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            type_ = item.get("@type", "")
            types = [type_] if isinstance(type_, str) else type_
            for t in types:
                if t in _SCHEMA_TYPE_MAP:
                    return ExtractedFact(
                        field_name=F_INDUSTRY,
                        value=_SCHEMA_TYPE_MAP[t],
                        source_url=source_url,
                        source_type=source_type,
                        confidence=0.90,
                    )
    return None


def _from_keywords(
    text: str, source_url: str, confidence_base: float = 0.75
) -> ExtractedFact | None:
    source_type = source_type_for_url(source_url)
    scores: dict[str, int] = {}
    for industry, patterns in _INDUSTRY_PATTERNS.items():
        hits = sum(1 for p in patterns if p.search(text))
        if hits > 0:
            scores[industry] = hits

    if not scores:
        return None

    best_industry = max(scores, key=lambda k: scores[k])
    hit_count = scores[best_industry]
    # Confidence scales with number of keyword hits, capped at 0.88
    confidence = min(confidence_base + (hit_count - 1) * 0.03, 0.88)

    return ExtractedFact(
        field_name=F_INDUSTRY,
        value=best_industry,
        source_url=source_url,
        source_type=source_type,
        confidence=confidence,
    )


def extract_industry(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Classify company industry from crawled pages.
    Returns at most one ExtractedFact — the highest-confidence classification.
    """
    best: ExtractedFact | None = None

    for page in pages:
        if not page.ok:
            continue
        try:
            soup = BeautifulSoup(page.html, "lxml")
        except Exception:
            continue

        source_type = source_type_for_url(page.url)

        # Schema.org wins if found
        schema_fact = _from_schema_org(soup, page.url)
        if schema_fact:
            if best is None or schema_fact.confidence > best.confidence:
                best = schema_fact

        # Entity-type-from-name detection (see _from_entity_type_name
        # docstring) — runs on every page since the title/og:site_name
        # tag is typically identical site-wide, but cheap to re-check.
        entity_fact = _from_entity_type_name(soup, page.url)
        if entity_fact:
            if best is None or entity_fact.confidence > best.confidence:
                best = entity_fact

        text = _text_from_page(soup)

        # Agency-phrase detection (see _from_agency_phrases docstring for
        # why this runs as its own pass rather than folded into the
        # generic keyword taxonomy below).
        agency_fact = _from_agency_phrases(text, page.url)
        if agency_fact:
            if best is None or agency_fact.confidence > best.confidence:
                best = agency_fact

        # Market-research/analyst-firm phrase detection (see
        # _from_market_research_phrases docstring).
        research_fact = _from_market_research_phrases(text, page.url)
        if research_fact:
            if best is None or research_fact.confidence > best.confidence:
                best = research_fact

        # Keyword taxonomy
        kw_fact = _from_keywords(text, page.url)
        if kw_fact:
            if best is None or kw_fact.confidence > best.confidence:
                best = kw_fact

    return [best] if best else []