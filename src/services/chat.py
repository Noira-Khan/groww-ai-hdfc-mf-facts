"""Shared question-processing service for the API and Streamlit frontends.

This module owns the guard order (PII -> intent -> safety routing -> RAG) so
both frontends behave identically. It never logs questions, answers, or
retrieved text.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from src.config import Config
from src.rag.pipeline import answer_question
from src.safety import responses
from src.safety.intent import SCHEME_LABELS, classify_question
from src.safety.pii import detect_pii

AnswerFn = Callable[..., dict[str, Any]]


class ChatUnavailableError(RuntimeError):
    """The factual pipeline is unavailable (e.g. missing corpus or index)."""


@dataclass(frozen=True)
class ChatResult:
    envelope: dict[str, Any]
    route: str
    request_id: str


def process_question(
    config: Config,
    question: str,
    scheme: str | None = None,
    *,
    model: Any = None,
    answer_fn: AnswerFn | None = None,
    request_id: str | None = None,
) -> ChatResult:
    """Return one safe response envelope for a question.

    The order of checks and the envelopes produced are identical to the
    original FastAPI handler. ``answer_fn`` is injectable so tests and callers
    can supply the same pipeline used by the API.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty")

    active_request_id = request_id or uuid.uuid4().hex[:16]
    answer = answer_fn or answer_question
    route = "answer"

    if config.safety.detect_pii and detect_pii(question):
        route = "pii"
        envelope = responses.pii_response(active_request_id)
    else:
        intent = classify_question(question, scheme)
        route = intent.route

        if intent.route == "refusal_advice":
            envelope = responses.advice_response(active_request_id)
        elif intent.route == "refusal_performance":
            envelope = responses.performance_response(active_request_id)
        elif intent.route == "not_in_sources":
            envelope = responses.not_in_sources_response(active_request_id)
        elif intent.route == "clarify":
            envelope = responses.clarify_response(question, active_request_id)
        else:
            answer_scheme = intent.first_scheme if intent.scheme_sensitive else None
            try:
                envelope = answer(
                    config,
                    question,
                    scheme=answer_scheme,
                    model=model,
                )
                if envelope.get("type") == "answer" and intent.extra_schemes:
                    envelope["can_do"] = [
                        f"What about {SCHEME_LABELS[item]}?"
                        for item in intent.extra_schemes
                    ]
            except ValueError as exc:
                raise ChatUnavailableError(str(exc)) from exc
            except Exception:
                envelope = responses.error_response(active_request_id)

    return ChatResult(
        envelope=envelope,
        route=route,
        request_id=active_request_id,
    )
