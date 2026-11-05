"""Lightweight topic detection for retrieval reranking and follow-ups."""

from __future__ import annotations

import re

from src.safety.intent import SCHEME_LABELS, SUPPORTED_SCHEMES

TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "exit_load",
        re.compile(r"\bexit[\s-]*load\b", re.IGNORECASE),
    ),
    (
        "expense_ratio",
        re.compile(r"\b(?:total\s+)?expense\s*ratio\b|\bter\b", re.IGNORECASE),
    ),
    (
        "min_sip",
        re.compile(
            r"\bmin(?:imum)?\s+(?:sip|investment|purchase|"
            r"lump[\s-]*sum|application)\b|"
            r"\bmin(?:imum)?\s+additional\s+purchase\b|"
            r"\bmin(?:imum)?\s+amount\b",
            re.IGNORECASE,
        ),
    ),
    (
        "riskometer",
        re.compile(
            r"\brisk[\s-]*o[\s-]*meter\b|\briskometer\b|"
            # Groww fund pages often omit "riskometer" and say "Very High Risk".
            r"(?:rated\s+)?(?:very\s+high|moderately\s+high|moderate|high|low)\s+risk\b",
            re.IGNORECASE,
        ),
    ),
    ("benchmark", re.compile(r"\bbenchmark(?:ing)?\b", re.IGNORECASE)),
    ("elss_lockin", re.compile(r"\block[\s-]*in\b|\blockin\b", re.IGNORECASE)),
    ("capital_gains", re.compile(r"\bcapital[\s-]*gains?\b", re.IGNORECASE)),
    (
        "statements",
        re.compile(r"\b(?:account\s+)?statement(?:s)?\b", re.IGNORECASE),
    ),
    (
        "definitions",
        re.compile(r"\bnav\b|\bnet\s+asset\s+value\b", re.IGNORECASE),
    ),
)

TOPIC_LABELS: dict[str, str] = {
    "exit_load": "exit load",
    "expense_ratio": "expense ratio",
    "min_sip": "minimum SIP",
    "riskometer": "riskometer",
    "benchmark": "benchmark",
    "elss_lockin": "lock-in period",
    "capital_gains": "capital-gains statement",
    "statements": "account statement",
    "definitions": "definition",
}

SCHEME_SENSITIVE_TOPICS = frozenset(
    {
        "exit_load",
        "expense_ratio",
        "min_sip",
        "riskometer",
        "benchmark",
        "elss_lockin",
    }
)

_TOPIC_PATTERN_MAP = {name: pattern for name, pattern in TOPIC_PATTERNS}


def detect_topic(question: str) -> str | None:
    """Return the first known fact topic mentioned in the question."""
    text = question or ""
    for topic, pattern in TOPIC_PATTERNS:
        if pattern.search(text):
            return topic
    return None


def text_matches_topic(topic: str, *parts: str) -> bool:
    """True when any text/section part mentions the detected topic."""
    pattern = _TOPIC_PATTERN_MAP.get(topic)
    if pattern is None:
        return False
    haystack = "\n".join(part for part in parts if part)
    if not haystack:
        return False
    return pattern.search(haystack) is not None


def scheme_topic_question(topic: str, scheme: str) -> str | None:
    """Build a self-contained follow-up question for another scheme."""
    label = SCHEME_LABELS.get(scheme)
    if scheme not in SUPPORTED_SCHEMES or not label:
        return None
    if topic not in SCHEME_SENSITIVE_TOPICS:
        return None
    return f"What is the {TOPIC_LABELS[topic]} of {label}?"
