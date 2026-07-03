"""
crawlers/extractors/hiring.py
──────────────────────────────
Detect hiring signals from crawled pages.

A company is considered to be actively hiring if:
  1. A /careers page exists and contains job listing patterns, OR
  2. Any page contains multiple job title mentions with apply language

Returns:
  - One F_HIRING_SIGNAL fact ("true") if active hiring detected
  - One F_JOB_TITLE fact per detected job title (up to 10)

Confidence:
  0.90 — careers page with explicit job listings
  0.80 — careers page exists but unclear if open roles
  0.70 — hiring language found on non-careers page
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from crawlers.extractors.base import (
    ExtractedFact,
    F_HIRING_SIGNAL,
    F_JOB_TITLE,
    SOURCE_CAREERS,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult

_MAX_JOB_TITLES = 10

# Patterns indicating active job listings
_APPLY_PATTERNS = re.compile(
    r"\b(apply now|apply today|submit.*resume|send.*resume|we.re hiring|"
    r"join our team|open position|current opening|job opening|"
    r"view opening|see opening|career opportunit|employment opportunit|"
    r"we are looking for|we.re looking for|now hiring)\b",
    re.I,
)

# Common job title patterns — broad enough to work across industries
_JOB_TITLE_PATTERNS = re.compile(
    r"\b("
    r"(?:senior|junior|lead|principal|associate|assistant|chief|head\s+of|"
    r"director\s+of|vp\s+of|manager\s+of|coordinator\s+of)?\s*"
    r"(?:software|web|mobile|frontend|backend|full.?stack|"
    r"marketing|sales|account|customer|support|"
    r"operations|product|project|program|"
    r"dental|medical|clinical|nursing|hygien|"
    r"legal|financial|accounting|hr|human\s+resources|"
    r"office|front\s+desk|receptionist|"
    r"graphic|ui|ux|design|creative|"
    r"data|analytics|research|"
    r"content|seo|social\s+media|"
    r"logistics|warehouse|driver|"
    r"construction|project|site)"
    r"\s*"
    r"(?:engineer|developer|manager|coordinator|specialist|analyst|"
    r"designer|director|assistant|associate|representative|"
    r"officer|administrator|consultant|advisor|lead|"
    r"hygienist|assistant|technician|receptionist|secretary|"
    r"attorney|paralegal|accountant|bookkeeper|"
    r"supervisor|superintendent|foreman)?"
    r")\b",
    re.I,
)


def extract_hiring(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Detect hiring signals and job titles from crawled pages.
    """
    facts: list[ExtractedFact] = []
    hiring_detected = False
    seen_titles: set[str] = set()

    for page in pages:
        if not page.ok:
            continue

        source_type = source_type_for_url(page.url)
        is_careers = source_type == SOURCE_CAREERS

        try:
            soup = BeautifulSoup(page.html, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
        except Exception:
            continue

        has_apply_language = bool(_APPLY_PATTERNS.search(text))

        if is_careers:
            confidence = 0.90 if has_apply_language else 0.80
            if not hiring_detected:
                hiring_detected = True
                facts.append(ExtractedFact(
                    field_name=F_HIRING_SIGNAL,
                    value="true",
                    source_url=page.url,
                    source_type=source_type,
                    confidence=confidence,
                ))
        elif has_apply_language and not hiring_detected:
            hiring_detected = True
            facts.append(ExtractedFact(
                field_name=F_HIRING_SIGNAL,
                value="true",
                source_url=page.url,
                source_type=source_type,
                confidence=0.70,
            ))

        # Extract job titles from careers page or any page with apply language
        if (is_careers or has_apply_language) and len(seen_titles) < _MAX_JOB_TITLES:
            for match in _JOB_TITLE_PATTERNS.finditer(text):
                title = " ".join(match.group(0).split()).title()
                title_lower = title.lower()
                if (
                    len(title.split()) >= 2
                    and title_lower not in seen_titles
                    and len(title) <= 60
                ):
                    seen_titles.add(title_lower)
                    facts.append(ExtractedFact(
                        field_name=F_JOB_TITLE,
                        value=title,
                        source_url=page.url,
                        source_type=source_type,
                        confidence=0.75 if is_careers else 0.65,
                    ))
                if len(seen_titles) >= _MAX_JOB_TITLES:
                    break

    return facts
