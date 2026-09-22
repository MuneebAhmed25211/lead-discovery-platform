"""
utils/text.py
─────────────
Text processing helpers used by extractors and scoring.

All functions are deterministic, pure Python — no LLM.
"""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalise unicode, strip HTML entities, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)       # HTML entities
    text = re.sub(r"<[^>]+>", " ", text)            # any remaining tags
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_emails(text: str) -> list[str]:
    """Extract all email addresses from a block of text."""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    return list(set(re.findall(pattern, text)))


def extract_phone_numbers(text: str) -> list[str]:
    """Extract common phone number formats."""
    pattern = r"(\+?\d[\d\s\-().]{7,}\d)"
    candidates = re.findall(pattern, text)
    # Filter out numbers that are clearly dates or zip codes
    return [c.strip() for c in candidates if len(re.sub(r"\D", "", c)) >= 7]


def keyword_overlap(text: str, keywords: list[str]) -> list[str]:
    """
    Return which keywords from the list appear in text (case-insensitive).
    Used by the scoring engine to measure ICP signal overlap.
    """
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


# Generic business-category words stripped before token-overlap comparison.
# NOT industry-specific — these are structural words that appear in almost
# any industry label regardless of vertical ("logistics COMPANIES", "legal
# FIRMS", "healthcare PROVIDERS"), so stripping them lets two differently
# phrased labels for the same real-world category match on their actual
# distinguishing word(s) instead of failing a whole-phrase substring check.
_GENERIC_CATEGORY_WORDS = frozenset({
    "company", "companies", "firm", "firms", "business", "businesses",
    "provider", "providers", "service", "services", "group", "groups",
    "and", "of", "the", "&", "inc", "llc", "ltd", "corp", "corporation",
})

# Short (<=2 char) tokens are normally stripped as noise (see below), but a
# few domain-relevant abbreviations carry real meaning at that length and
# would otherwise silently vanish from every comparison — e.g. an ICP
# positive_signal like "job posting for machine learning or AI engineer"
# would lose the one word ("AI") that most specifically identifies the
# signal.
_MEANINGFUL_SHORT_TOKENS = frozenset({"ai", "ml"})


def significant_tokens(phrase: str) -> set[str]:
    """
    Tokenize a short category phrase (e.g. an industry name) into its
    meaningful words, stripping generic business-structure words and
    tokens under 3 characters (except _MEANINGFUL_SHORT_TOKENS). Used for
    comparing two differently-worded labels for what may be the same
    real-world category — e.g. "logistics companies" vs "Logistics &
    Supply Chain" both reduce to a set containing "logistics", which a
    whole-phrase substring check would miss entirely (neither string
    contains the other verbatim).
    """
    words = re.findall(r"[a-z0-9]+", phrase.lower())
    return {
        w for w in words
        if w not in _GENERIC_CATEGORY_WORDS
        and (len(w) > 2 or w in _MEANINGFUL_SHORT_TOKENS)
    }


def significant_token_overlap(a: str, b: str) -> set[str]:
    """Return the set of significant tokens shared between two phrases."""
    return significant_tokens(a) & significant_tokens(b)


# Synonym groups for industry-label matching specifically. Deliberately
# NOT folded into significant_tokens() itself — that function is also used
# by vendor-disqualifier detection and ICP pain-point/signal matching
# (see call sites in scoring_service.py), and expanding synonyms there
# could change behavior in ways that haven't been checked. This is scoped
# to industry_token_overlap() only.
#
# Confirmed bug this fixes: a deterministically-extracted industry like
# "Legal" and an ICP target_industries entry like "law firms" share zero
# literal tokens ("legal" != "law"), even though they're the same
# real-world category — significant_token_overlap() alone can bridge
# differently-WORDED phrasings of the same words (e.g. "logistics
# companies" vs "Logistics & Supply Chain", both containing "logistics"),
# but not true synonyms where the words themselves differ. Anchored to
# the same taxonomy as crawlers/extractors/industry.py's _INDUSTRY_KEYWORDS
# so the two stay in sync.
_INDUSTRY_SYNONYMS: dict[str, frozenset[str]] = {
    "legal": frozenset({"law", "lawyer", "lawyers", "attorney", "attorneys", "litigation"}),
    "healthcare": frozenset({"medical", "clinic", "clinics", "physician", "physicians",
                              "doctor", "doctors", "hospital", "hospitals"}),
    "dental": frozenset({"dentist", "dentists", "orthodontic", "orthodontics"}),
    "accounting": frozenset({"accountant", "accountants", "cpa", "bookkeeping", "tax", "taxes"}),
    "retail": frozenset({"store", "stores", "shop", "shops"}),
    "construction": frozenset({
        "contractor", "contractors", "remodeling", "renovation",
        # Added: confirmed gap — ICP target phrasings for the specific
        # home-services trades industry.py's own taxonomy already routes
        # into "Construction & Home Services" (see _INDUSTRY_KEYWORDS'
        # "Construction & Home Services" list and the Plumber/Electrician/
        # GeneralContractor _SCHEMA_TYPE_MAP entries) shared zero tokens
        # with "construction" itself, so those companies scored 0/25 on
        # industry_fit despite being correctly classified. Verified via
        # utils.text.industry_token_overlap("Construction & Home Services",
        # "plumbing and HVAC companies") returning an empty set before
        # this fix.
        "plumbing", "plumber", "plumbers",
        "hvac", "heating", "cooling",
        "landscaping", "landscaper", "landscapers", "lawn", "lawncare",
        "roofing", "roofer", "roofers",
        "electrical", "electrician", "electricians",
        "handyman", "handymen",
    }),
    "consulting": frozenset({"consultant", "consultants", "advisory"}),
    "staffing": frozenset({"recruiting", "recruitment", "recruiter", "recruiters"}),
    # Added alongside the "Events & Weddings" industry category fix in
    # crawlers/extractors/industry.py. Bridges to ICP phrasing like
    # "wedding planning firms" and "small event planning agencies".
    "events": frozenset({"event", "wedding", "weddings", "bridal",
                          "planner", "planners", "planning"}),
    # Added alongside the "Hospitality" industry category fix in
    # crawlers/extractors/industry.py. Bridges to ICP phrasing like
    # "boutique hotels" and "family-owned bed and breakfast establishments".
    "hospitality": frozenset({"hotel", "hotels", "inn", "inns", "lodge",
                                "lodges", "resort", "resorts", "lodging",
                                "guesthouse", "guesthouses", "bnb"}),
    "manufacturing": frozenset({"manufacturer", "manufacturers", "factory", "industrial"}),
    "insurance": frozenset({"insurer", "insurers"}),
    # Same gap class as "construction" above — "Beauty & Wellness" (the
    # taxonomy label) shares no literal token with how ICPs phrase this
    # vertical ("beauty salons and spas", "day spas", "nail salons").
    "beauty": frozenset({"salon", "salons", "spa", "spas", "cosmetic",
                          "cosmetics", "nail", "nails", "esthetics"}),
    # Same gap class as "construction"/"beauty" above — "Banking &
    # Financial Services" (the taxonomy label) shares no literal token
    # with common ICP phrasings ("financial services firms", "banks",
    # "credit unions").
    "banking": frozenset({"bank", "banks", "banking", "financial",
                           "finance", "credit", "union", "unions",
                           "lender", "lenders", "lending"}),
    # Added alongside the "Automotive" industry category in
    # crawlers/extractors/industry.py. Bridges the single-word extracted
    # label "Automotive" to how ICPs actually phrase this vertical (e.g.
    # "auto repair shops", "vehicle service chains", "auto body repair
    # centers") — none of which share the literal token "automotive".
    "automotive": frozenset({"auto", "autos", "vehicle", "vehicles", "car",
                              "cars", "dealership", "dealerships",
                              "fleet", "fleets"}),
}


def _expand_industry_synonyms(tokens: set[str]) -> set[str]:
    """Expand a token set with any matching synonym groups, both directions."""
    expanded = set(tokens)
    for token in tokens:
        for canonical, synonyms in _INDUSTRY_SYNONYMS.items():
            if token == canonical or token in synonyms:
                expanded.add(canonical)
                expanded |= synonyms
    return expanded


def industry_token_overlap(a: str, b: str) -> set[str]:
    """
    Like significant_token_overlap(), but also bridges known industry
    synonyms (e.g. "Legal" <-> "law firms") that don't share a literal
    token. Used only for industry_fit scoring — see module docstring
    above _INDUSTRY_SYNONYMS for why this isn't folded into
    significant_tokens() itself.
    """
    ta = _expand_industry_synonyms(significant_tokens(a))
    tb = _expand_industry_synonyms(significant_tokens(b))
    return ta & tb


def signal_overlap_hits(text: str, signals: list[str], min_shared_tokens: int = 2) -> list[str]:
    """
    Return which ICP-authored signal sentences (pain_points, positive_signals,
    disqualifying_signals, etc.) are substantively evidenced in `text`.

    Full-sentence substring matching (see keyword_overlap) essentially never
    fires for these fields: ICP signals are free-form Groq-generated
    sentences ("job posting for machine learning or AI engineer") that
    practically never appear verbatim in extracted company evidence
    ("Analytics Lead, Associate Clinical..."). This instead requires at
    least `min_shared_tokens` significant tokens in common — enough to
    mean something, not so little that any single incidental word overlap
    (e.g. one company mentioning "service") counts as a match.
    """
    text_tokens = significant_tokens(text)
    if not text_tokens:
        return []
    hits = []
    for signal in signals:
        shared = text_tokens & significant_tokens(signal)
        if len(shared) >= min_shared_tokens:
            hits.append(signal)
    return hits


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Return the number of unique keywords found in text."""
    return len(keyword_overlap(text, keywords))


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars, appending ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def sentences(text: str, max_sentences: int = 3) -> str:
    """Return the first N sentences of text."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:max_sentences])