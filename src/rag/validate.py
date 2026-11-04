"""Deterministic validation for generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

from src.config import Config
from src.rag.retrieve import RetrievedChunk

ALLOWED_URL_HOSTS = ("hdfcfund.com", "sebi.gov.in", "amfiindia.com")

URL_RE = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
TRAILING_URL_CHARS = ".,;:!?)]}"
NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
ABBREVIATION_RE = re.compile(
    r"\b(?:Rs|Dr|Mr|Mrs|Ms|No|approx|e\.g|i\.e|vs|p\.a)\.", re.IGNORECASE
)
PII_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    re.compile(r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)"),
    re.compile(r"\b(?:otp|one[- ]time password)\b", re.IGNORECASE),
)
ADVICE_PATTERNS = (
    r"\byou should\b",
    r"\b(?:we|i) recommend\b",
    r"\brecommend(?:ed|ing|ation)?\b",
    r"\bshould (?:buy|sell|invest|hold|switch|redeem)\b",
    r"\bbuy or sell\b",
    r"\bgood investment\b",
    r"\bworth investing\b",
    r"\binvest now\b",
    r"\bbook (?:profit|loss)\b",
    r"\bguaranteed\b",
    r"\bbetter (?:fund|scheme|option|choice)\b",
    r"\bbest (?:fund|scheme|option|choice)\b",
    r"\binvestment advice\b",
)
PERFORMANCE_PATTERNS = (
    r"\breturns?\b",
    r"\bperformance\b",
    r"\bcagr\b",
    r"\bannuali[sz]ed\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\boutperform\b",
    r"\brank(?:ed|ing)?\b",
    r"\bpast\b",
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    answer: str = ""
    source_url: str = ""
    source_name: str = ""
    as_on_date: str | None = None
    retrieved_at: str | None = None
    proof: str = ""
    not_in_sources: bool = False


def _strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in URL_RE.finditer(text):
        urls.append(match.group(0).rstrip(TRAILING_URL_CHARS))
    return urls


def count_sentences(text: str) -> int:
    cleaned = _strip_urls(text).strip()
    if not cleaned:
        return 0
    cleaned = ABBREVIATION_RE.sub(lambda match: match.group(0).replace(".", "\u2024"), cleaned)
    cleaned = re.sub(r"(?<=\d)\.(?=\d)", "\u2024", cleaned)
    parts = [part for part in re.split(r"[.!?]+(?=\s|$)", cleaned) if part.strip()]
    return len(parts) or 1


def _canonical_number(token: str) -> str:
    raw = token.replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return raw
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _number_tokens(text: str) -> set[str]:
    return {_canonical_number(match.group(0)) for match in NUMBER_RE.finditer(text)}


def _host_is_allowed(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_URL_HOSTS)


def is_allowlisted_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and _host_is_allowed(parsed.hostname or "")


def _contains_any(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def _context_text(question: str, chunks: list[RetrievedChunk]) -> str:
    parts = [question]
    for chunk in chunks:
        parts.append(chunk.text)
        for key in ("section", "scheme", "topic", "publisher"):
            value = chunk.metadata.get(key)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def validate_candidate(
    config: Config,
    candidate: dict,
    chunks: list[RetrievedChunk],
    question: str,
) -> ValidationResult:
    """Run the deterministic answer checks in the required order."""
    if candidate.get("not_in_sources"):
        return ValidationResult(ok=False, not_in_sources=True)

    issues: list[ValidationIssue] = []
    answer = str(candidate.get("answer") or "").strip()
    source_url = str(candidate.get("source_url") or "").strip()
    proof = str(candidate.get("proof") or "").strip()

    if not answer:
        issues.append(ValidationIssue("answer_missing", "Answer is empty."))
    else:
        sentence_count = count_sentences(answer)
        if sentence_count > config.app.max_answer_sentences:
            issues.append(
                ValidationIssue(
                    "too_many_sentences",
                    f"Answer has {sentence_count} sentences; maximum is "
                    f"{config.app.max_answer_sentences}.",
                )
            )

    answer_urls = extract_urls(answer)
    urls = list(dict.fromkeys(answer_urls + ([source_url] if source_url else [])))
    if len(urls) != 1:
        issues.append(
            ValidationIssue(
                "url_count",
                f"Response must carry exactly one URL; found {len(urls)}.",
            )
        )

    if not source_url:
        issues.append(ValidationIssue("url_missing", "source_url is missing."))
    elif answer_urls and source_url not in answer_urls:
        issues.append(
            ValidationIssue(
                "url_not_in_answer",
                "source_url must be the single URL used in the answer.",
            )
        )

    if source_url:
        if not is_allowlisted_url(source_url):
            issues.append(
                ValidationIssue("url_not_allowlisted", "source_url is not an official URL.")
            )
        match = next((chunk for chunk in chunks if chunk.url == source_url), None)
        if match is None:
            issues.append(
                ValidationIssue(
                    "url_not_retrieved",
                    "source_url was not present in retrieved chunk metadata.",
                )
            )
        else:
            publisher = str(match.metadata.get("publisher") or "")
            if publisher not in config.ingestion.allowlist_publishers:
                issues.append(
                    ValidationIssue(
                        "publisher_not_allowlisted",
                        "source_url publisher is not allowlisted.",
                    )
                )

    banned = _contains_any(answer, ADVICE_PATTERNS)
    if banned:
        issues.append(
            ValidationIssue("advice_phrase", "Answer contains an advice-like phrase.")
        )
    banned = _contains_any(answer, PERFORMANCE_PATTERNS)
    if banned:
        issues.append(
            ValidationIssue(
                "performance_phrase",
                "Answer contains a performance-comparison phrase.",
            )
        )

    answer_numbers = _number_tokens(_strip_urls(answer))
    context_numbers = _number_tokens(_context_text(question, chunks))
    ungrounded = sorted(answer_numbers - context_numbers)
    if ungrounded:
        issues.append(
            ValidationIssue(
                "ungrounded_number",
                "Answer numbers absent from context: " + ", ".join(ungrounded) + ".",
            )
        )

    if not proof:
        issues.append(ValidationIssue("proof_missing", "proof is empty."))
    else:
        cited = next((chunk for chunk in chunks if chunk.url == source_url), None)
        pool = [cited] if cited is not None else chunks
        if not any(proof in chunk.text for chunk in pool):
            issues.append(
                ValidationIssue(
                    "proof_not_verbatim",
                    "proof is not a verbatim substring of the cited retrieved chunk.",
                )
            )

    if _contains_any(answer, tuple(pattern.pattern for pattern in PII_PATTERNS)):
        issues.append(
            ValidationIssue("pii_echo", "Answer contains a PII-like value.")
        )

    cited = next((chunk for chunk in chunks if chunk.url == source_url), None)
    source_name = str(cited.metadata.get("publisher") or "") if cited else ""
    as_on_date = cited.metadata.get("as_on_date") if cited else None
    retrieved_at = None
    if cited:
        retrieved_at = cited.metadata.get("retrieved_at") or cited.metadata.get("fetched_at")

    return ValidationResult(
        ok=not issues,
        issues=issues,
        answer=answer,
        source_url=source_url,
        source_name=source_name,
        as_on_date=str(as_on_date) if as_on_date else None,
        retrieved_at=str(retrieved_at) if retrieved_at else None,
        proof=proof,
        not_in_sources=False,
    )
