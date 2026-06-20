"""
utils/fingerprint.py
────────────────────
Browser fingerprint validation utilities.

FingerprintJS Pro visitor IDs are 32-character hex strings.
We also accept shorter IDs (min 8 chars) from other libraries
so the system works during development without a Pro subscription.

The fingerprint is NOT personally identifiable information —
it's a probabilistic device identifier derived from browser signals.
We treat it as pseudonymous and store only its hashed form in analytics.
"""

from __future__ import annotations

import hashlib
import re

# FingerprintJS Pro: 32-char hex. Community edition varies.
_VALID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")


def is_valid_fingerprint(fp: str) -> bool:
    """
    Return True if the fingerprint string looks legitimate.

    Accepts:
      - 8–64 character strings
      - alphanumeric plus hyphen and underscore
    Rejects:
      - Empty strings
      - Strings with spaces or special characters
      - Strings under 8 chars (too short to be meaningful)
    """
    if not fp:
        return False
    return bool(_VALID_PATTERN.match(fp))


def hash_fingerprint(fp: str) -> str:
    """
    Return a one-way SHA-256 hash of the fingerprint.

    Used when storing fingerprints in analytics logs where we want
    to count unique visitors without storing the raw identifier.
    """
    return hashlib.sha256(fp.encode("utf-8")).hexdigest()[:16]
