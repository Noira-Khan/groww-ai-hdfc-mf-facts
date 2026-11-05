"""End-to-end retrieval, generation, validation, and caching."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable

from src.config import Config
from src.rag.cache import get_cached_answer, put_cached_answer
from src.rag.generate import GenerationError, generate_candidate
from src.rag.retrieve import RetrievalResult, retrieve
from src.rag.validate import validate_candidate
from src.safety.intent import SCHEME_LABELS, classify_question
from src.safety.responses import pii_response
from src.suggestions import SAFE_QUESTIONS

SCHEME_DOCUMENTS_URL = "https://www.hdfcfund.com/mutual-funds/fund-documents/kim"


def _focus_question(
    question: str, first_scheme: str | None, extra_schemes: list[str]
) -> str:
    """Keep only the first scheme so generation can answer it, not refuse all."""
    if not first_scheme or not extra_schemes:
        return question
    focused = question
    for scheme in extra_schemes:
        label = SCHEME_LABELS.get(scheme)
        if not label:
            continue
        focused = re.sub(
            rf"\s*(?:,|and|or|&)?\s*{re.escape(label)}",
            "",
            focused,
            flags=re.IGNORECASE,
        )
    focused = re.sub(r"\s+(?:and|or)\s*(?=[?.!]|$)", "", focused, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", focused).strip()


def _not_in_sources_envelope(request_id: str) -> dict[str, Any]:
    return {
        "type": "not_in_sources",
        "answer": "That fact is not covered by my curated official sources.",
        "source_url": SCHEME_DOCUMENTS_URL,
        "source_name": "HDFC AMC",
        "as_on_date": None,
        "proof": None,
        "can_do": list(SAFE_QUESTIONS),
        "chunks_used": [],
        "request_id": request_id,
    }


def _error_envelope(request_id: str) -> dict[str, Any]:
    return {
        "type": "error",
        "answer": "I could not produce a verified answer right now. Please try again.",
        "source_url": None,
        "source_name": None,
        "as_on_date": None,
        "proof": None,
        "can_do": [],
        "chunks_used": [],
        "request_id": request_id,
    }


def _generate_with_retry(
    config: Config,
    question: str,
    chunks,
    **kwargs,
):
    """Retry provider and parsing failures once before surfacing an error."""
    last_error: GenerationError | None = None
    for _ in range(config.safety.max_repair_retries + 1):
        try:
            return generate_candidate(config, question, chunks, **kwargs)
        except GenerationError as exc:
            last_error = exc
    raise last_error or GenerationError("Provider failed without an error.")


def answer_question(
    config: Config,
    question: str,
    scheme: str | None = None,
    *,
    model=None,
    complete_fn: Callable[[Config, list[dict[str, str]]], str] | None = None,
    persist_debug: bool = True,
) -> dict[str, Any]:
    """Return one validated response envelope for a factual question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty")

    cached = get_cached_answer(config, question, scheme)
    if cached is not None:
        return cached

    intent = classify_question(question, scheme)
    retrieval_scheme = scheme or (
        intent.first_scheme if intent.scheme_sensitive else None
    )
    generation_question = _focus_question(
        question, intent.first_scheme, intent.extra_schemes
    )

    retrieval: RetrievalResult = retrieve(
        config,
        question,
        scheme=retrieval_scheme,
        model=model,
        persist_debug=persist_debug,
    )
    request_id = retrieval.debug["request_id"]

    if not retrieval.chunks:
        return _not_in_sources_envelope(request_id)

    detected_topic = retrieval.debug.get("detected_topic")
    if detected_topic:
        chunk_topics = {
            chunk.metadata.get("topic")
            for chunk in retrieval.chunks
            if chunk.metadata.get("topic")
        }
        if chunk_topics and detected_topic not in chunk_topics:
            return _not_in_sources_envelope(request_id)

    try:
        generation = _generate_with_retry(
            config,
            generation_question,
            retrieval.chunks,
            complete_fn=complete_fn,
        )
    except GenerationError:
        return _error_envelope(request_id)

    result = validate_candidate(
        config, generation.payload, retrieval.chunks, generation_question
    )
    if result.not_in_sources:
        return _not_in_sources_envelope(request_id)

    retries = config.safety.max_repair_retries
    while not result.ok and retries > 0:
        try:
            generation = _generate_with_retry(
                config,
                generation_question,
                retrieval.chunks,
                repair_candidate=generation.raw,
                repair_issues=[issue.message for issue in result.issues],
                complete_fn=complete_fn,
            )
        except GenerationError:
            return _error_envelope(request_id)
        result = validate_candidate(
            config, generation.payload, retrieval.chunks, generation_question
        )
        retries -= 1

    if not result.ok:
        if any(issue.code == "pii_echo" for issue in result.issues):
            return pii_response(request_id)
        return _not_in_sources_envelope(request_id)

    envelope = {
        "type": "answer",
        "answer": result.answer,
        "source_url": result.source_url,
        "source_name": result.source_name,
        "as_on_date": result.as_on_date,
        "retrieved_at": result.retrieved_at or date.today().isoformat(),
        "proof": result.proof,
        "can_do": [],
        "chunks_used": [chunk.chunk_id for chunk in retrieval.chunks],
        "request_id": request_id,
    }
    put_cached_answer(config, question, scheme, envelope)
    return envelope
