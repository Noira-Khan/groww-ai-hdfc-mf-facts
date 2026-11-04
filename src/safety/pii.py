"""Detect personal data before any model call."""

from __future__ import annotations

import re

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    ("aadhaar", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9](?:[\s-]?\d){9}(?!\d)"),
    ),
    (
        "otp",
        re.compile(
            r"\b(?:otp|one[\s-]?time\s*password)\b[^\d]{0,16}\d{4,8}\b|"
            r"\b\d{4,8}\b[^\d]{0,16}\b(?:otp|one[\s-]?time\s*password)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account",
        re.compile(
            r"\b(?:folio|account)(?:\s*(?:no\.?|number|#))?"
            r"\s*(?:(?:is|:|-)\s*)?\d{4,}\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_pii(text: str) -> str | None:
    """Return the PII category found, or None. Never log the matched value."""
    if not text:
        return None
    for category, pattern in PII_PATTERNS:
        if pattern.search(text):
            return category
    return None
