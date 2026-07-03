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
        "counsel", "esquire", "llp", "legal advice", "paralegal",
        "immigration", "criminal defense", "personal injury", "corporate law",
        "intellectual property", "patent", "trademark",
    ],
    "Accounting & Finance": [
        "accounting", "accountant", "cpa", "bookkeeping", "tax preparation",
        "audit", "financial planning", "wealth management", "investment",
        "payroll", "controller", "cfo", "financial advisor", "tax filing",
        "budgeting", "forensic accounting",
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
        "flooring", "painting", "home improvement", "building",
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
        "assembly", "industrial", "oem", "iso certified", "quality control",
        "lean manufacturing", "cnc",
    ],
    "Insurance": [
        "insurance", "insurer", "underwriting", "claims", "policy",
        "life insurance", "health insurance", "property insurance",
        "risk management", "reinsurance",
    ],
}


def _text_from_page(soup: BeautifulSoup, max_chars: int = 5000) -> str:
    """Extract readable text from a page, capped to avoid regex slowdowns."""
    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "header"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:max_chars].lower()


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
    for industry, keywords in _INDUSTRY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
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

        # Keyword taxonomy
        text = _text_from_page(soup)
        kw_fact = _from_keywords(text, page.url)
        if kw_fact:
            if best is None or kw_fact.confidence > best.confidence:
                best = kw_fact

    return [best] if best else []
