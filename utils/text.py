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
