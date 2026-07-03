"""
crawlers/extractors/technologies.py
────────────────────────────────────
Detect technologies used by a company from their website.

Approach: fingerprint matching against HTML source — script src attributes,
meta generator tags, link rel attributes, inline JS variable names, and
known CSS class patterns. This is deterministic — no LLM.

Returns one ExtractedFact per detected technology.
Technologies are useful Phase 5 scoring signals:
  - "Salesforce" → they have a CRM, probably don't need a basic one
  - "Calendly"   → they use online booking, might want automation
  - "HubSpot"    → marketing-mature company
  - "WordPress"  → non-technical team, open to turnkey solutions
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crawlers.extractors.base import (
    ExtractedFact,
    F_TECHNOLOGY,
    source_type_for_url,
)
from crawlers.p1_crawler import PageResult


@dataclass(frozen=True)
class TechSignature:
    """A single technology fingerprint."""
    name: str
    pattern: re.Pattern
    confidence: float


def _sig(name: str, pattern: str, confidence: float = 0.85) -> TechSignature:
    return TechSignature(name=name, pattern=re.compile(pattern, re.I), confidence=confidence)


# Fingerprint library — match against the full HTML source (not just text)
# Patterns are matched against the raw HTML string for performance
_SIGNATURES: list[TechSignature] = [
    # CMS
    _sig("WordPress", r"wp-content|wp-includes|wordpress", 0.90),
    _sig("Wix", r"wix\.com|static\.wixstatic\.com", 0.90),
    _sig("Squarespace", r"squarespace\.com|static1\.squarespace", 0.90),
    _sig("Webflow", r"webflow\.com|webflow\.io", 0.90),
    _sig("Shopify", r"cdn\.shopify\.com|shopify\.com/s/", 0.90),
    _sig("Weebly", r"weebly\.com|editmysite\.com", 0.85),
    _sig("GoDaddy Website Builder", r"godaddysites\.com|secureserver\.net/wp", 0.85),

    # CRM / Marketing platforms
    _sig("HubSpot", r"hs-scripts\.com|hubspot\.com|hbspt\.", 0.90),
    _sig("Salesforce", r"salesforce\.com|force\.com|pardot\.com", 0.90),
    _sig("ActiveCampaign", r"activecampaign\.com|activehosted\.com", 0.85),
    _sig("Mailchimp", r"mailchimp\.com|chimpstatic\.com|list-manage\.com", 0.85),
    _sig("Klaviyo", r"klaviyo\.com|static\.klaviyo\.com", 0.85),
    _sig("Zoho CRM", r"zoho\.com|zohostatic\.com", 0.80),
    _sig("Pipedrive", r"pipedrive\.com", 0.80),
    _sig("Intercom", r"intercom\.io|widget\.intercom\.io", 0.85),
    _sig("Drift", r"drift\.com|js\.driftt\.com", 0.85),
    _sig("Zendesk", r"zendesk\.com|zdassets\.com", 0.85),

    # Booking / Scheduling
    _sig("Calendly", r"calendly\.com|assets\.calendly\.com", 0.90),
    _sig("Acuity Scheduling", r"acuityscheduling\.com", 0.90),
    _sig("Mindbody", r"mindbodyonline\.com|booker\.com", 0.90),
    _sig("Zocdoc", r"zocdoc\.com/widget|zocdoc-btn", 0.90),
    _sig("Jane App", r"jane\.app|janeapp\.com", 0.90),
    _sig("Healthie", r"healthie\.com|gethealthie\.com", 0.85),
    _sig("Square Appointments", r"squareup\.com/appointments", 0.85),
    _sig("Booksy", r"booksy\.com", 0.85),
    _sig("OpenTable", r"opentable\.com/widget", 0.85),
    _sig("SimplyBook", r"simplybook\.me|simplybook\.it", 0.85),

    # Analytics
    _sig("Google Analytics", r"google-analytics\.com|gtag\(|ga\.js|analytics\.js", 0.85),
    _sig("Google Tag Manager", r"googletagmanager\.com", 0.80),
    _sig("Facebook Pixel", r"connect\.facebook\.net|fbq\(", 0.85),
    _sig("Hotjar", r"hotjar\.com|static\.hotjar\.com", 0.80),
    _sig("Mixpanel", r"mixpanel\.com|cdn\.mxpnl\.com", 0.80),

    # Payment
    _sig("Stripe", r"stripe\.com/v3|js\.stripe\.com", 0.85),
    _sig("Square", r"squareup\.com|square-cdn\.com", 0.80),
    _sig("PayPal", r"paypal\.com/sdk|paypalobjects\.com", 0.80),

    # Live chat
    _sig("Tidio", r"tidio\.com|tidiochat\.com", 0.85),
    _sig("LiveChat", r"livechatinc\.com|cdn\.livechatinc\.com", 0.85),
    _sig("Tawk.to", r"tawk\.to|embed\.tawk\.to", 0.85),

    # Review platforms
    _sig("Birdeye", r"birdeye\.com|embedsocial\.com/birdeye", 0.80),
    _sig("Podium", r"podium\.com|reviews\.podium\.com", 0.80),

    # Practice management (dental/medical specific but harmless for others)
    _sig("Dentrix", r"dentrix\.com|dentrixascend", 0.85),
    _sig("Eaglesoft", r"eaglesoft\.net|eaglesoft", 0.85),
    _sig("Open Dental", r"opendental\.com", 0.85),
    _sig("Curve Dental", r"curvedental\.com", 0.85),
    _sig("Kareo", r"kareo\.com", 0.85),
    _sig("Athenahealth", r"athenahealth\.com|athenanet\.com", 0.85),
    _sig("drchrono", r"drchrono\.com", 0.85),
]


def extract_technologies(pages: list[PageResult]) -> list[ExtractedFact]:
    """
    Detect technologies from raw HTML of crawled pages.
    Returns one ExtractedFact per unique detected technology.
    """
    detected: dict[str, ExtractedFact] = {}

    for page in pages:
        if not page.ok:
            continue
        source_type = source_type_for_url(page.url)
        html = page.html

        for sig in _SIGNATURES:
            if sig.name in detected:
                continue
            if sig.pattern.search(html):
                detected[sig.name] = ExtractedFact(
                    field_name=F_TECHNOLOGY,
                    value=sig.name,
                    source_url=page.url,
                    source_type=source_type,
                    confidence=sig.confidence,
                )

    return list(detected.values())
