"""
crawlers/extractors/booking.py
────────────────────────────────
Detect online booking widget presence.

This is a strong scoring signal for many ICPs: a company with an
existing booking widget is more likely to want AI-powered scheduling
automation than one still taking bookings by phone.

Detection methods:
  1. Known booking platform script/iframe signatures in raw HTML
  2. "Book online" / "Schedule online" / "Book appointment" CTA patterns
  3. Embedded iframe src pointing to booking platforms

Returns one F_BOOKING_WIDGET fact if any booking signal is detected.
"""

from __future__ import annotations

import re

from crawlers.extractors.base import (
    ExtractedFact,
    F_BOOKING_WIDGET,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult


# Raw HTML signatures for booking platforms
_BOOKING_SIGNATURES: list[tuple[str, re.Pattern]] = [
    ("Calendly",         re.compile(r"calendly\.com", re.I)),
    ("Acuity",           re.compile(r"acuityscheduling\.com", re.I)),
    ("Mindbody",         re.compile(r"mindbodyonline\.com|booker\.com", re.I)),
    ("Zocdoc Widget",    re.compile(r"zocdoc\.com/widget|zocdoc-btn", re.I)),
    ("Jane App",         re.compile(r"jane\.app|janeapp\.com", re.I)),
    ("Healthie",         re.compile(r"gethealthie\.com|healthie\.com", re.I)),
    ("Square Appts",     re.compile(r"squareup\.com/appointments", re.I)),
    ("Booksy",           re.compile(r"booksy\.com", re.I)),
    ("OpenTable",        re.compile(r"opentable\.com/widget", re.I)),
    ("SimplyBook",       re.compile(r"simplybook\.me|simplybook\.it", re.I)),
    ("Setmore",          re.compile(r"setmore\.com", re.I)),
    ("Vagaro",           re.compile(r"vagaro\.com", re.I)),
    ("Fresha",           re.compile(r"fresha\.com", re.I)),
    ("10to8",            re.compile(r"10to8\.com", re.I)),
    ("YouCanBook.me",    re.compile(r"youcanbook\.me", re.I)),
    ("Doodle",           re.compile(r"doodle\.com/en/embed", re.I)),
    ("TimeTrade",        re.compile(r"timetrade\.com", re.I)),
    ("HubSpot Meetings", re.compile(r"meetings\.hubspot\.com", re.I)),
    ("Chili Piper",      re.compile(r"chilipiper\.com", re.I)),
    ("Generic iframe",   re.compile(r'<iframe[^>]+(?:booking|schedule|appointment|reserve)', re.I)),
]

# CTA text patterns that strongly imply booking capability
_BOOKING_CTA = re.compile(
    r"\b(book\s+(?:online|now|an?\s+appointment)|"
    r"schedule\s+(?:online|now|an?\s+appointment|a\s+visit|a\s+consultation)|"
    r"request\s+an?\s+appointment|"
    r"reserve\s+(?:online|now|your\s+spot)|"
    r"online\s+booking|"
    r"book\s+a\s+(?:demo|call|session|class|service))\b",
    re.I,
)


def extract_booking(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Detect booking widget presence from crawled pages.
    Returns at most one ExtractedFact.
    """
    for page in pages:
        if not page.ok:
            continue

        source_type = source_type_for_url(page.url)
        html = page.html

        # Check raw HTML for platform signatures (highest confidence)
        for platform_name, pattern in _BOOKING_SIGNATURES:
            if pattern.search(html):
                return [ExtractedFact(
                    field_name=F_BOOKING_WIDGET,
                    value=platform_name,
                    source_url=page.url,
                    source_type=source_type,
                    confidence=0.90,
                )]

        # Check for CTA text (lower confidence — might just be a phone booking CTA)
        if _BOOKING_CTA.search(html):
            return [ExtractedFact(
                field_name=F_BOOKING_WIDGET,
                value="online_booking_cta",
                source_url=page.url,
                source_type=source_type,
                confidence=0.65,
            )]

    return []
