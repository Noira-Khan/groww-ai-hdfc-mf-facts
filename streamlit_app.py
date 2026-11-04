"""Streamlit frontend for the HDFC MF Facts assistant.

The app shell renders immediately. Configuration is loaded quickly, while the
embedding model and knowledge base are initialized only when needed, with
visible stage-level progress. Answer generation still uses the shared
``src.services.chat.process_question`` path, so all safety controls and
validation behave exactly like the FastAPI app.
"""

from __future__ import annotations

import os
import time
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
from src.logging import get_logger, safe_log
from src.safety.intent import SCHEME_LABELS
from src.services.kb_state import artifacts_present

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

logger = get_logger("streamlit")


@dataclass
class Runtime:
    config: Config | None = None
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


@st.cache_resource(show_spinner=False)
def get_runtime() -> Runtime:
    """Load configuration only. This must stay fast so the shell renders early."""
    _apply_secrets()

    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:
        return Runtime(error=f"Configuration could not be loaded: {exc}")

    try:
        warnings = tuple(validate_runtime(config))
    except Exception as exc:
        return Runtime(config=config, error=f"Configuration is invalid: {exc}")

    return Runtime(config=config, warnings=warnings)


@st.cache_resource(show_spinner=False)
def get_model() -> Any:
    """Load and cache the embedding model without blocking the initial shell."""
    runtime = get_runtime()
    if runtime.config is None:
        raise RuntimeError("Configuration is not available.")

    from src.ingest.embed import load_model

    return load_model(runtime.config)


@st.cache_resource(show_spinner=False)
def get_knowledge_base() -> bool:
    """Validate the Chroma collection once and cache the result."""
    runtime = get_runtime()
    if runtime.config is None:
        return False

    from src.services.knowledge_base import collection_ready

    return collection_ready(runtime.config)


def _source_link(url: str, name: str | None) -> None:
    from src.rag.validate import is_allowlisted_url

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
        item for item in questions[:6] if isinstance(item, str) and item.strip()
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
                message.get("envelope")
                or {
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
            (code for code, label in SCHEME_LABELS.items() if label == choice),
            None,
        )

        st.divider()
        st.markdown("**Official sources**")
        st.markdown(
            "- [HDFC AMC fund documents](https://www.hdfcfund.com/mutual-funds/fund-documents/kim)\n"
            "- [SEBI investor education](https://investor.sebi.gov.in/)\n"
            "- [AMFI knowledge center](https://www.amfiindia.com/investor/knowledge-center-info)"
        )

        if runtime.config is not None:
            st.divider()
            st.markdown("**Knowledge base**")
            if artifacts_present(runtime.config):
                st.caption("Initialized. Source-backed answers are enabled.")
            else:
                st.caption("Not initialized yet. Use the setup panel to build it.")

        if runtime.warnings:
            st.divider()
            st.warning(
                "Deployment notes:\n\n"
                + "\n".join(f"- {warning}" for warning in runtime.warnings)
            )

        st.divider()
        st.caption(DISCLAIMER)


def _render_config_error(runtime: Runtime) -> None:
    st.error("The assistant is not configured correctly.")
    if runtime.error:
        st.markdown(f"**Details:** {runtime.error}")
    if st.button("Reload configuration"):
        st.cache_resource.clear()
        st.rerun()


def _render_setup(config: Config) -> None:
    st.warning(
        "The official knowledge base has not been built in this deployment yet. "
        "It is created once from the curated HDFC AMC, SEBI, and AMFI sources."
    )
    st.markdown(
        "**What happens when you initialize:**\n"
        "1. Load the local embedding model.\n"
        "2. Download the official documents, with live progress.\n"
        "3. Normalize and chunk the documents.\n"
        "4. Embed the chunks and build the Chroma index.\n"
        "5. Validate the index before the assistant can answer."
    )
    st.caption(
        "The first run can take several minutes. The page stays responsive and "
        "each stage reports its own progress."
    )
    if st.button("Initialize knowledge base", type="primary", key="initialize_kb"):
        _run_initialization(config, force=False)


def _run_initialization(config: Config, force: bool = False) -> None:
    from src.services.knowledge_base import KnowledgeBaseError, ensure_knowledge_base

    progress_bar = st.progress(0.0, text="Starting knowledge-base initialization…")
    status = st.status("Initializing the knowledge base…", expanded=True)
    last_stage = {"name": None}

    def on_progress(stage: str, fraction: float, message: str) -> None:
        safe_fraction = max(0.0, min(1.0, float(fraction)))
        progress_bar.progress(safe_fraction, text=message)
        status.update(label=message)
        if stage != last_stage["name"]:
            status.write(message)
            last_stage["name"] = stage

    try:
        on_progress("model", 0.02, "Loading the embedding model…")
        model_started = time.perf_counter()
        model = get_model()
        safe_log(
            logger,
            "kb_stage",
            stage="model",
            status="ok",
            latency_ms=int((time.perf_counter() - model_started) * 1000),
        )
        on_progress("model", 0.12, "Embedding model ready.")

        def kb_progress(stage: str, fraction: float, message: str) -> None:
            on_progress(stage, 0.12 + 0.88 * fraction, message)

        ensure_knowledge_base(
            config, model=model, progress=kb_progress, force=force
        )
    except KnowledgeBaseError as exc:
        status.update(label="Knowledge-base initialization failed", state="error")
        st.error(str(exc))
        st.info("Check the deployment logs for the failing stage, then retry.")
        return
    except Exception as exc:
        status.update(label="Knowledge-base initialization failed", state="error")
        st.error(f"Knowledge-base initialization failed unexpectedly. ({exc})")
        return

    status.update(label="Knowledge base ready", state="complete")
    progress_bar.progress(1.0, text="Knowledge base ready.")
    get_knowledge_base.clear()
    st.rerun()


def _render_chat_input(runtime: Runtime) -> None:
    from src.services.chat import ChatUnavailableError, process_question

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
    kb_unavailable = False
    with st.spinner("Checking the official documents…"):
        if not get_knowledge_base():
            kb_unavailable = True
        else:
            try:
                model = get_model()
                result = process_question(
                    runtime.config,
                    question,
                    scheme,
                    model=model,
                )
            except ChatUnavailableError:
                kb_unavailable = True
            except ValueError:
                st.warning("Please enter a question.")
                return
            except Exception:
                st.error(
                    "The assistant could not process that question right now. "
                    "Please try again."
                )
                return

    if kb_unavailable:
        st.error(
            "The knowledge base is unavailable right now. "
            "Please try again in a moment."
        )
        if st.button("Repair knowledge base", key="repair_kb"):
            _run_initialization(runtime.config, force=True)
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

    if runtime.config is None or runtime.error:
        _render_config_error(runtime)
        return

    st.info(DISCLAIMER)
    _render_history()

    if not artifacts_present(runtime.config):
        _render_setup(runtime.config)
        return

    _render_chat_input(runtime)


main()
