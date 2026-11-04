"""Deterministic routing for factual, refusal, and clarification cases."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SUPPORTED_SCHEMES = ("large_cap", "flexi_cap", "elss", "mid_cap")
SCHEME_LABELS: dict[str, str] = {
    "large_cap": "HDFC Large Cap Fund",
    "flexi_cap": "HDFC Flexi Cap Fund",
    "elss": "HDFC ELSS Tax Saver",
    "mid_cap": "HDFC Mid Cap Fund",
    "small_cap": "HDFC Small Cap Fund",
}
SCHEME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("large_cap", re.compile(r"\blarge[\s-]*cap\b", re.IGNORECASE)),
    ("flexi_cap", re.compile(r"\bflexi[\s-]*cap\b", re.IGNORECASE)),
    ("elss", re.compile(r"\belss\b|\btax[\s-]*saver\b", re.IGNORECASE)),
    ("mid_cap", re.compile(r"\bmid[\s-]*cap\b", re.IGNORECASE)),
    ("small_cap", re.compile(r"\bsmall[\s-]*cap\b", re.IGNORECASE)),
)
SCHEME_SENSITIVE_TOPICS = re.compile(
    r"\bexit\s*load\b|\bexpense\s*ratio\b|\bter\b|\bminimum\s+(?:sip|investment|"
    r"lump\s*sum)\b|\briskometer\b|\bbenchmark\b|\block[\s-]*in\b",
    re.IGNORECASE,
)
INJECTION_PATTERNS = re.compile(
    r"\bignore\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions?|rules?)\b|"
    r"\bsystem\s+prompt\b|\bjailbreak\b|\bdeveloper\s+mode\b|"
    r"\bpretend\s+(?:you\s+are|to\s+be)\b|\breveal\s+(?:your\s+)?(?:prompt|rules)\b",
    re.IGNORECASE,
)
PERFORMANCE_PATTERNS = re.compile(
    r"\breturns?\b|\bperformance\b|\bcagr\b|\bcompare\b|\bcomparison\b|"
    r"\boutperform\b|\brank(?:ed|ing)?\b|\bbest\b.*\b(?:return|gain)\b|"
    r"\bwhich\s+fund\s+gave\b",
    re.IGNORECASE,
)
ADVICE_PATTERNS = re.compile(
    r"\bshould\s+i\s+(?:buy|sell|invest|hold|switch|redeem)\b|"
    r"\bwhich\s+(?:is|one\s+is)\s+better\b|"
    r"\bwhich\s+(?:fund|scheme|option|plan)\s+is\s+(?:better|best)\b|"
    r"\b(?:is|are)\s+[^?.!]{0,80}?\bbetter\s+than\b|"
    r"\bbetter\s+(?:fund|scheme|option)\b|"
    r"\brecommend(?:ed|ation)?\b|\bgood\s+investment\b|\bworth\s+(?:investing|it)\b|"
    r"\binvest\s+in\b|\bportfolio\b|\bbuy\s+or\s+sell\b|"
    r"\bwhere\s+should\s+i\s+invest\b|\bbest\s+fund\b",
    re.IGNORECASE,
)

AMC_NAMES = (
    r"hdfc|sbi|icici|axis|kotak|nippon|dsp|franklin|tata|uti|mirae|parag|"
    r"quant|motilal|aditya\s+birla|hsbc|bandhan|canara|lic|idfc|edelweiss|"
    r"invesco|pgim|sundaram|baroda|navi|360\s+one|jio|white\s*oak"
)
NAMED_FUND_PATTERN = re.compile(
    rf"\b(?:{AMC_NAMES})\b[^?.!]{{0,80}}?\b(?:fund|scheme)\b", re.IGNORECASE
)
SUPPORTED_FUND_NAME = re.compile(
    r"\b(?:large|flexi|mid)[\s-]*cap\b|\belss\b|\btax[\s-]*saver\b", re.IGNORECASE
)


def mentions_unsupported_fund(question: str) -> bool:
    """True when a named fund/scheme is outside the curated corpus."""
    for match in NAMED_FUND_PATTERN.finditer(question):
        if not SUPPORTED_FUND_NAME.search(match.group(0)):
            return True
    return False


@dataclass
class IntentResult:
    route: str
    schemes: list[str] = field(default_factory=list)
    scheme_sensitive: bool = False

    @property
    def first_scheme(self) -> str | None:
        return self.schemes[0] if self.schemes else None

    @property
    def extra_schemes(self) -> list[str]:
        return self.schemes[1:]


def detect_schemes(question: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for scheme, pattern in SCHEME_PATTERNS:
        match = pattern.search(question)
        if match:
            found.append((match.start(), scheme))
    ordered: list[str] = []
    for _, scheme in sorted(found):
        if scheme not in ordered:
            ordered.append(scheme)
    return ordered


def classify_question(question: str, scheme: str | None = None) -> IntentResult:
    schemes = detect_schemes(question)
    if scheme and scheme not in schemes:
        schemes.insert(0, scheme)

    if INJECTION_PATTERNS.search(question):
        return IntentResult("not_in_sources", schemes)

    if PERFORMANCE_PATTERNS.search(question):
        return IntentResult("refusal_performance", schemes)

    if ADVICE_PATTERNS.search(question):
        return IntentResult("refusal_advice", schemes)

    if mentions_unsupported_fund(question):
        return IntentResult("not_in_sources", schemes)

    scheme_sensitive = bool(SCHEME_SENSITIVE_TOPICS.search(question))
    unsupported = [item for item in schemes if item not in SUPPORTED_SCHEMES]
    if unsupported:
        return IntentResult("not_in_sources", schemes, scheme_sensitive)

    if scheme_sensitive and not schemes:
        return IntentResult("clarify", schemes, True)

    return IntentResult("answer", schemes, scheme_sensitive)
