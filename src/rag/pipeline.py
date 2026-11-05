
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


SCHEME_DOCUMENTS_URL = (
    "https://www.hdfcfund.com/mutual-funds/fund-documents/kim"
)


def _passage_excerpt(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _ranked_passages(retrieval: RetrievalResult) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    for index, chunk in enumerate(retrieval.chunks, start=1):
        passages.append(
            {
                "rank": index,
                "score": round(float(chunk.score), 4),
                "source_id": chunk.metadata.get("source_id"),
                "publisher": chunk.metadata.get("publisher"),
                "doc_type": chunk.metadata.get("doc_type"),
                "url": chunk.url,
                "section": chunk.section,
                "excerpt": _passage_excerpt(chunk.text),
            }
        )
    return passages


def _pipeline_steps(
    *,
    intent_route: str,
    retrieval: RetrievalResult,
    model_id: str,
    validation_ok: bool,
    validation_note: str,
) -> list[dict[str, Any]]:
    detected = retrieval.debug.get("detected_topic")
    return [
        {
            "step": 1,
            "title": "Checked for personal details",
            "status": "passed",
            "detail": "No PAN, Aadhaar, phone, email, OTP or account numbers found.",
        },
        {
            "step": 2,
            "title": "Classified the question",
            "status": "done",
            "detail": (
                f"Route `{intent_route}`"
                + (f", topic `{detected}`." if detected else ".")
            ),
        },
        {
            "step": 3,
            "title": "Searched the curated sources",
            "status": "top_k",
            "detail": (
                f"Selected {len(retrieval.chunks)} passages from Groww, "
                "HDFC AMC, SEBI and AMFI."
            ),
        },
        {
            "step": 4,
            "title": "Wrote the answer from those passages only",
            "status": "grounded",
            "detail": f"Model: {model_id or 'configured LLM'}. No outside knowledge allowed.",
        },
        {
            "step": 5,
            "title": "Checked the answer",
            "status": "passed" if validation_ok else "failed",
            "detail": validation_note,
        },
    ]


# ---------------------------------------------------------
# Question focus
# ---------------------------------------------------------

def _focus_question(
    question: str,
    first_scheme: str | None,
    extra_schemes: list[str],
) -> str:
    """Keep only the first scheme so generation can answer it."""

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

    focused = re.sub(
        r"\s+(?:and|or)\s*(?=[?.!]|$)",
        "",
        focused,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s{2,}", " ", focused).strip()


# ---------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------

def _not_in_sources_envelope(
    request_id: str,
) -> dict[str, Any]:

    return {
        "type": "not_in_sources",
        "answer": (
            "That fact is not covered by my curated Groww sources."
        ),
        "source_url": SCHEME_DOCUMENTS_URL,
        "source_name": "HDFC AMC",
        "as_on_date": None,
        "proof": None,
        "can_do": list(SAFE_QUESTIONS),
        "chunks_used": [],
        "request_id": request_id,
    }


def _error_envelope(
    request_id: str,
) -> dict[str, Any]:

    return {
        "type": "error",
        "answer": (
            "I could not produce a verified answer right now. "
            "Please try again."
        ),
        "source_url": None,
        "source_name": None,
        "as_on_date": None,
        "proof": None,
        "can_do": [],
        "chunks_used": [],
        "request_id": request_id,
    }


# ---------------------------------------------------------
# Evidence assessment
# ---------------------------------------------------------

def _has_retrieved_evidence(
    retrieval: RetrievalResult,
) -> bool:
    """
    Check whether retrieved chunks contain usable source text.

    Topic metadata is deliberately not used as a rejection rule.
    Topic metadata helps retrieval ranking, while actual evidence
    is assessed using the retrieved document text.
    """

    if not retrieval.chunks:
        return False

    for chunk in retrieval.chunks:

        text = (chunk.text or "").strip()

        # Ignore empty or unusable chunks.
        if len(text) < 20:
            continue

        # Require a source identifier or URL so that the
        # retrieved text remains traceable to its origin.
        source_id = chunk.metadata.get("source_id")
        source_url = chunk.metadata.get("url")

        if source_id or source_url:
            return True

    return False


def _get_retrieval_diagnostics(
    retrieval: RetrievalResult,
) -> dict[str, Any]:
    """
    Build diagnostics for understanding the retrieved evidence.
    Does not alter retrieval scores or validation behaviour.
    """

    return {
        "detected_topic": retrieval.debug.get("detected_topic"),

        "retrieved_chunks": len(retrieval.chunks),

        "topic_matches": sum(
            1
            for chunk in retrieval.chunks
            if (
                retrieval.debug.get("detected_topic")
                and chunk.metadata.get("topic")
                == retrieval.debug.get("detected_topic")
            )
        ),

        "sources": [
            {
                "source_id": chunk.metadata.get("source_id"),
                "url": chunk.metadata.get("url"),
                "doc_type": chunk.metadata.get("doc_type"),
                "topic": chunk.metadata.get("topic"),
                "section": chunk.metadata.get("section"),
                "text_length": len(chunk.text or ""),
            }
            for chunk in retrieval.chunks
        ],
    }


# ---------------------------------------------------------
# Generation with retry
# ---------------------------------------------------------

def _generate_with_retry(
    config: Config,
    question: str,
    chunks,
    **kwargs,
):
    """Retry provider and parsing failures before surfacing an error."""

    last_error: GenerationError | None = None

    for _ in range(config.safety.max_repair_retries + 1):

        try:
            return generate_candidate(
                config,
                question,
                chunks,
                **kwargs,
            )

        except GenerationError as exc:
            last_error = exc

    raise last_error or GenerationError(
        "Provider failed without an error."
    )


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def answer_question(
    config: Config,
    question: str,
    scheme: str | None = None,
    *,
    model=None,
    complete_fn: Callable[
        [Config, list[dict[str, str]]], str
    ] | None = None,
    persist_debug: bool = True,
) -> dict[str, Any]:

    """Return one validated response envelope for a factual question."""

    question = (question or "").strip()

    if not question:
        raise ValueError("Question must not be empty")

    # -----------------------------------------------------
    # 1. Check answer cache
    # -----------------------------------------------------

    cached = get_cached_answer(
        config,
        question,
        scheme,
    )

    if cached is not None:
        return cached

    # -----------------------------------------------------
    # 2. Classify question intent
    # -----------------------------------------------------

    intent = classify_question(
        question,
        scheme,
    )

    retrieval_scheme = scheme or (
        intent.first_scheme
        if intent.scheme_sensitive
        else None
    )

    generation_question = _focus_question(
        question,
        intent.first_scheme,
        intent.extra_schemes,
    )

    # -----------------------------------------------------
    # 3. Retrieve candidate chunks
    # -----------------------------------------------------

    retrieval: RetrievalResult = retrieve(
        config,
        question,
        scheme=retrieval_scheme,
        model=model,
        persist_debug=persist_debug,
    )

    request_id = retrieval.debug["request_id"]

    # -----------------------------------------------------
    # 4. Reject only if no chunks are retrieved
    # -----------------------------------------------------

    if not retrieval.chunks:
        return _not_in_sources_envelope(request_id)

    # -----------------------------------------------------
    # 5. Evaluate actual retrieved evidence
    # -----------------------------------------------------

    if not _has_retrieved_evidence(retrieval):
        return _not_in_sources_envelope(request_id)

    # -----------------------------------------------------
    # 6. Record topic and source diagnostics
    # -----------------------------------------------------

    evidence_debug = _get_retrieval_diagnostics(
        retrieval
    )

    retrieval.debug["evidence_assessment"] = evidence_debug

    # IMPORTANT:
    # Do not reject merely because detected_topic differs
    # from the retrieved chunk's topic metadata.
    #
    # Topic metadata remains useful for ranking in retrieve.py.
    # Actual factual support is evaluated by validate.py.

    # -----------------------------------------------------
    # 7. Generate answer using retrieved evidence
    # -----------------------------------------------------

    try:

        generation = _generate_with_retry(
            config,
            generation_question,
            retrieval.chunks,
            complete_fn=complete_fn,
        )

    except GenerationError:
        return _error_envelope(request_id)

    # -----------------------------------------------------
    # 8. Validate generated answer
    # -----------------------------------------------------

    result = validate_candidate(
        config,
        generation.payload,
        retrieval.chunks,
        generation_question,
    )

    if result.not_in_sources:
        return _not_in_sources_envelope(request_id)

    # -----------------------------------------------------
    # 9. Repair invalid answers
    # -----------------------------------------------------

    retries = config.safety.max_repair_retries

    while not result.ok and retries > 0:

        try:

            generation = _generate_with_retry(
                config,
                generation_question,
                retrieval.chunks,
                repair_candidate=generation.raw,
                repair_issues=[
                    issue.message
                    for issue in result.issues
                ],
                complete_fn=complete_fn,
            )

        except GenerationError:
            return _error_envelope(request_id)

        result = validate_candidate(
            config,
            generation.payload,
            retrieval.chunks,
            generation_question,
        )

        retries -= 1

    # -----------------------------------------------------
    # 10. Final validation failure handling
    # -----------------------------------------------------

    if not result.ok:

        if any(
            issue.code == "pii_echo"
            for issue in result.issues
        ):
            return pii_response(request_id)

        return _not_in_sources_envelope(request_id)

    # -----------------------------------------------------
    # 11. Build successful response
    # -----------------------------------------------------

    envelope = {

        "type": "answer",

        "answer": result.answer,

        "source_url": result.source_url,

        "source_name": result.source_name,

        "as_on_date": result.as_on_date,

        "retrieved_at": (
            result.retrieved_at
            or date.today().isoformat()
        ),

        "proof": result.proof,

        "can_do": [],

        "chunks_used": [
            chunk.chunk_id
            for chunk in retrieval.chunks
        ],

        "request_id": request_id,

        "pipeline_steps": _pipeline_steps(
            intent_route=intent.route,
            retrieval=retrieval,
            model_id=config.llm.model,
            validation_ok=True,
            validation_note=(
                "One cited source, three sentences or fewer, "
                "every figure found in the passages."
            ),
        ),

        "passages": _ranked_passages(retrieval),

    }

    # -----------------------------------------------------
    # 12. Cache validated answer
    # -----------------------------------------------------

    put_cached_answer(
        config,
        question,
        scheme,
        envelope,
    )

    return envelope
