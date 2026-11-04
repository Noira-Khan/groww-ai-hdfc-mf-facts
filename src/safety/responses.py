"""Fixed, safe response envelopes for guardrail routes."""

from __future__ import annotations

from typing import Any

from src.safety.intent import SCHEME_LABELS, SUPPORTED_SCHEMES

STATEMENT_URL = "https://www.hdfcfund.com/services/additional-info/request-statement"
SCHEME_DOCUMENTS_URL = "https://www.hdfcfund.com/mutual-funds/fund-documents/kim"
FACTSHEET_URL = "https://www.hdfcfund.com/mutual-funds/factsheets"
ADVICE_EDUCATION_URL = "https://investor.sebi.gov.in/"

SAFE_QUESTIONS = [
    "What is the exit load of HDFC Flexi Cap Fund?",
    "What is the lock-in period of HDFC ELSS Tax Saver?",
    "How do I download my capital-gains statement?",
]


def envelope(
    response_type: str,
    answer: str,
    *,
    source_url: str | None = None,
    source_name: str | None = None,
    as_on_date: str | None = None,
    proof: str | None = None,
    can_do: list[str] | None = None,
    chunks_used: list[str] | None = None,
    request_id: str,
) -> dict[str, Any]:
    return {
        "type": response_type,
        "answer": answer,
        "source_url": source_url,
        "source_name": source_name,
        "as_on_date": as_on_date,
        "proof": proof,
        "can_do": can_do or [],
        "chunks_used": chunks_used or [],
        "request_id": request_id,
    }


def pii_response(request_id: str) -> dict[str, Any]:
    return envelope(
        "pii",
        "Please do not share personal information such as PAN, Aadhaar, phone, "
        "email, OTP, or folio numbers.",
        source_url=STATEMENT_URL,
        source_name="HDFC AMC",
        can_do=list(SAFE_QUESTIONS),
        request_id=request_id,
    )


def advice_response(request_id: str) -> dict[str, Any]:
    return envelope(
        "refusal_advice",
        "I can share factual information, not investment advice or recommendations.",
        source_url=ADVICE_EDUCATION_URL,
        source_name="SEBI",
        can_do=list(SAFE_QUESTIONS),
        request_id=request_id,
    )


def performance_response(request_id: str) -> dict[str, Any]:
    return envelope(
        "refusal_performance",
        "I do not compute, compare, or comment on returns or performance.",
        source_url=FACTSHEET_URL,
        source_name="HDFC AMC",
        can_do=list(SAFE_QUESTIONS),
        request_id=request_id,
    )


def not_in_sources_response(request_id: str) -> dict[str, Any]:
    return envelope(
        "not_in_sources",
        "That fact is not covered by my curated official sources.",
        source_url=SCHEME_DOCUMENTS_URL,
        source_name="HDFC AMC",
        can_do=list(SAFE_QUESTIONS),
        request_id=request_id,
    )


def clarify_response(question: str, request_id: str) -> dict[str, Any]:
    base = question.strip().rstrip(" ?.")
    questions = [f"{base} of {SCHEME_LABELS[scheme]}?" for scheme in SUPPORTED_SCHEMES]
    return envelope(
        "clarify",
        "Which HDFC scheme do you mean? Choose one to continue.",
        source_url=SCHEME_DOCUMENTS_URL,
        source_name="HDFC AMC",
        can_do=questions,
        request_id=request_id,
    )


def error_response(request_id: str) -> dict[str, Any]:
    return envelope(
        "error",
        "I could not produce a verified answer right now. Please try again.",
        request_id=request_id,
    )
