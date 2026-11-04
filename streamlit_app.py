"""Streamlit frontend for the HDFC MF Facts assistant.

The UI reuses the same guard order and validation path as the FastAPI app via
``src.services.chat.process_question``. The Chroma index is opened when it is
already valid, and built only when it is missing or unusable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="HDFC MF Facts",
    layout="centered",
    initial_sidebar_state="expanded",
)

from src.config import Config, load_config, validate_runtime
from src.ingest.embed import load_model
from src.rag.validate import is_allowlisted_url
from src.safety.intent import SCHEME_LABELS
from src.services.chat import ChatUnavailableError, process_question
from src.services.knowledge_base import KnowledgeBaseError, ensure_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

SECRET_ENV_KEYS = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "LLM_MODEL_ID",
    "LLM_PROVIDER",
    "CACHE_SALT",
    "HF_TOKEN",
)

TYPE_LABELS = {
    "answer": "ANSWER FROM OFFICIAL SOURCES",
    "refusal_advice": "FACTS-ONLY ASSISTANT",
    "refusal_performance": "FACTS-ONLY ASSISTANT",
    "not_in_sources": "OUTSIDE CURATED SOURCES",
    "clarify": "CHOOSE A SCHEME",
    "pii": "PRIVACY PROTECTION",
    "error": "SOMETHING WENT WRONG",
}

TYPE_TAGS = {
    "refusal_advice": "No advice or recommendations.",
    "refusal_performance": "No return or performance views.",
    "not_in_sources": "Only curated official documents are used.",
    "clarify": "Scheme context is required for this fact.",
    "pii": "Personal data is not shown, logged, or stored.",
    "error": "No unverified answer is shown.",
}

DISCLAIMER = (
    "This assistant shares factual information from official HDFC AMC, SEBI, "
    "and AMFI sources only. It does not provide investment advice, "
    "recommendations, or performance comparisons. Mutual fund investments are "
    "subject to market risks; read all scheme-related documents carefully."
)


@dataclass
class Runtime:
    config: Config | None = None
    model: Any = None
    ready: bool = False
    error: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _secret_value(name: str) -> str | None:
    try:
        value = st.secrets[name]
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _apply_secrets() -> None:
    """Copy deploy-time secrets into the environment before config loading."""
    for name in SECRET_ENV_KEYS:
        value = _secret_value(name)
        if value:
            os.environ[name] = value


@st.cache_resource(show_spinner="Preparing the official knowledge base…")
def get_runtime() -> Runtime:
    _apply_secrets()

    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:
        return Runtime(error=f"Configuration could not be loaded: {exc}")

    try:
        warnings = tuple(validate_runtime(config))
    except Exception as exc:
        return Runtime(config=config, error=f"Configuration is invalid: {exc}")

    try:
        model = load_model(config)
    except Exception as exc:
        return Runtime(
            config=config,
            warnings=warnings,
            error=(
                "The embedding model could not be loaded. Check the deployment "
                f"logs and network access to Hugging Face. ({exc})"
            ),
        )

    try:
        ensure_knowledge_base(config, model)
    except KnowledgeBaseError as exc:
        return Runtime(
            config=config,
            model=model,
            warnings=warnings,
            error=str(exc),
        )
    except Exception as exc:
        return Runtime(
            config=config,
            model=model,
            warnings=warnings,
            error=f"Knowledge-base initialization failed unexpectedly. ({exc})",
        )

    return Runtime(config=config, model=model, ready=True, warnings=warnings)


def _source_link(url: str, name: str | None) -> None:
    label = name or "Official source"
    if is_allowlisted_url(url):
        st.markdown(f"**Official source:** [{label}]({url})")
    else:
        st.markdown("**Official source:**")
        st.text(url)


def _render_answer_body(envelope: dict[str, Any]) -> None:
    response_type = str(envelope.get("type") or "error")
    answer = str(envelope.get("answer") or "")

    st.caption(TYPE_LABELS.get(response_type, "RESPONSE"))
    if response_type == "answer":
        st.markdown(answer)
    elif response_type == "clarify":
        st.info(answer)
    elif response_type == "error":
        st.error(answer)
    else:
        st.warning(answer)

    tag = TYPE_TAGS.get(response_type)
    if tag:
        st.caption(tag)

    source_url = envelope.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        _source_link(source_url.strip(), envelope.get("source_name"))

    source_date = envelope.get("as_on_date") or envelope.get("retrieved_at")
    st.caption(f"Source date: {source_date or 'Not provided'}")

    proof = envelope.get("proof")
    if isinstance(proof, str) and proof.strip():
        st.markdown("**Verbatim proof passage**")
        st.code(proof, language=None)


def _queue_followup(question: str) -> None:
    st.session_state.pending_question = question
    st.session_state.pending_auto_scheme = True
    st.rerun()


def _render_followups(questions: Any, key_prefix: str) -> None:
    if not isinstance(questions, list) or not questions:
        return
    safe_questions = [
        item
        for item in questions[:6]
        if isinstance(item, str) and item.strip()
    ]
    if not safe_questions:
        return
    st.caption("Explore official facts")
    for index, question in enumerate(safe_questions):
        if st.button(question, key=f"{key_prefix}_followup_{index}"):
            _queue_followup(question)


def _render_assistant(
    envelope: dict[str, Any], key_prefix: str, show_followups: bool
) -> None:
    with st.chat_message("assistant"):
        _render_answer_body(envelope)
        if show_followups:
            _render_followups(envelope.get("can_do"), key_prefix)


def _render_history() -> None:
    for index, message in enumerate(st.session_state.messages):
        if message.get("role") == "user":
            with st.chat_message("user"):
                st.markdown(message.get("display") or "")
        elif message.get("role") == "assistant":
            is_last = index == len(st.session_state.messages) - 1
            _render_assistant(
                message.get("envelope") or {
                    "type": "error",
                    "answer": "I could not produce a verified answer right now.",
                },
                key_prefix=f"message_{index}",
                show_followups=is_last,
            )


def _sidebar(runtime: Runtime) -> None:
    with st.sidebar:
        st.header("HDFC MF Facts")
        st.caption("Facts-only. No investment advice.")
        st.markdown(
            "Ask factual questions about HDFC Large Cap, Flexi Cap, ELSS Tax "
            "Saver, and Mid Cap schemes."
        )

        st.divider()
        options = ["Auto-detect"] + list(SCHEME_LABELS.values())
        choice = st.selectbox(
            "Scheme context",
            options,
            index=0,
            help=(
                "Optional. Scheme-sensitive questions can also name a scheme. "
                "The curated corpus covers Large Cap, Flexi Cap, ELSS, and Mid Cap."
            ),
        )
        st.session_state.selected_scheme = next(
            (
                code
                for code, label in SCHEME_LABELS.items()
                if label == choice
            ),
            None,
        )

        st.divider()
        st.markdown("**Official sources**")
        st.markdown(
            "- [HDFC AMC fund documents](https://www.hdfcfund.com/mutual-funds/fund-documents/kim)\n"
            "- [SEBI investor education](https://investor.sebi.gov.in/)\n"
            "- [AMFI knowledge center](https://www.amfiindia.com/investor/knowledge-center-info)"
        )

        if runtime.warnings:
            st.divider()
            st.warning(
                "Deployment notes:\n\n"
                + "\n".join(f"- {warning}" for warning in runtime.warnings)
            )

        st.divider()
        st.caption(DISCLAIMER)


def _render_error(runtime: Runtime) -> None:
    st.error("The assistant is not ready to answer yet.")
    if runtime.error:
        st.markdown(f"**Details:** {runtime.error}")
    st.markdown(
        "The corpus and Chroma index are generated artifacts and are not "
        "committed to source control. They are built automatically on first "
        "run when they are missing."
    )
    st.code("python -m src.ingest --stage all", language="powershell")
    if st.button("Retry initialization"):
        get_runtime.clear()
        st.rerun()


def main() -> None:
    st.title("HDFC MF Facts")
    st.caption("Mutual fund facts, made clear. Facts-only. No investment advice.")
    st.markdown(
        "Ask about exit load, lock-in, minimum SIP, expense ratio, riskometer, "
        "benchmark, or account statements. Every factual answer cites one "
        "official source, its date, and a verbatim proof passage."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "pending_auto_scheme" not in st.session_state:
        st.session_state.pending_auto_scheme = False
    if "selected_scheme" not in st.session_state:
        st.session_state.selected_scheme = None

    runtime = get_runtime()
    _sidebar(runtime)

    if not runtime.ready:
        _render_error(runtime)
        st.stop()

    st.info(DISCLAIMER)

    _render_history()

    question = st.chat_input(
        "Ask a factual question about HDFC Mutual Fund schemes…"
    )
    use_auto_scheme = False
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        use_auto_scheme = st.session_state.pending_auto_scheme
        st.session_state.pending_question = None
        st.session_state.pending_auto_scheme = False

    if not question:
        return

    if len(question) > 2000:
        st.warning("Please keep questions under 2000 characters.")
        return

    scheme = None if use_auto_scheme else st.session_state.selected_scheme
    with st.spinner("Checking the official documents…"):
        try:
            result = process_question(
                runtime.config,
                question,
                scheme,
                model=runtime.model,
            )
        except ChatUnavailableError:
            st.error(
                "The knowledge base is unavailable right now. "
                "Please try again in a moment."
            )
            return
        except ValueError:
            st.warning("Please enter a question.")
            return

    envelope = result.envelope
    display_question = (
        "[Personal information hidden]"
        if envelope.get("type") == "pii"
        else question
    )
    st.session_state.messages.append(
        {"role": "user", "display": display_question}
    )
    st.session_state.messages.append(
        {"role": "assistant", "envelope": envelope}
    )

    with st.chat_message("user"):
        st.markdown(display_question)
    _render_assistant(
        envelope,
        key_prefix=f"message_{len(st.session_state.messages) - 1}",
        show_followups=True,
    )


main()
