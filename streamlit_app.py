"""Streamlit frontend for the HDFC MF Facts assistant.

The app shell renders immediately. Configuration is loaded quickly, while the
embedding model and knowledge base are initialized only when needed, with
visible stage-level progress. Suggested questions prefill the editable composer.

Answer generation uses ``src.services.chat.process_question``, so safety
controls and validation match the FastAPI app.

Deploy notes (Streamlit Community Cloud):
- Main file: ``streamlit_app.py``
- Python version: 3.13 (Advanced settings)
- Secrets: ``GROQ_API_KEY``, ``LLM_MODEL_ID`` (see ``.streamlit/secrets.toml.example``)
- First cold start builds the corpus; use **Initialize knowledge base**
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="HDFC MF Facts",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from src.config import Config, load_config, validate_runtime
from src.logging import get_logger, safe_log
from src.services.kb_state import artifacts_present
from src.suggestions import DEFAULT_SUGGESTIONS

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
    "clarify": "SPECIFY A SCHEME",
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
    "This assistant shares factual information from official HDFC AMC, SEBI, and "
    "AMFI sources only. It does not provide investment advice, recommendations, "
    "or performance comparisons. Mutual fund investments are subject to market "
    "risks; read all scheme-related documents carefully."
)
GITHUB_URL = "https://github.com/Noira-Khan/groww-ai-hdfc-mf-facts"
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


# Apply Streamlit secrets as early as possible so load_config / provider checks
# see the same environment on Community Cloud as local .env does.
_apply_secrets()


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


@st.cache_data(show_spinner=False)
def get_suggested_questions() -> tuple[str, ...]:
    """Load factual questions verified against the current corpus."""
    runtime = get_runtime()
    if runtime.config is None:
        return DEFAULT_SUGGESTIONS
    from src.rag.suggestions import resolve_suggestions

    verified = tuple(resolve_suggestions(runtime.config, limit=8))
    return verified or DEFAULT_SUGGESTIONS


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
    """Prefill the editable composer with a suggested question."""
    st.session_state["_clear_question_input"] = False
    st.session_state["question_input"] = question
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
    """Show the latest exchange immediately and keep older turns collapsible."""
    messages = st.session_state.messages

    def render_message(index: int, message: dict[str, Any]) -> None:
        if message.get("role") == "user":
            with st.chat_message("user"):
                st.markdown(message.get("display") or "")
        elif message.get("role") == "assistant":
            _render_assistant(
                message.get("envelope")
                or {
                    "type": "error",
                    "answer": "I could not produce a verified answer right now.",
                },
                key_prefix=f"message_{index}",
                show_followups=index == len(messages) - 1,
            )

    if len(messages) > 2:
        older_messages = messages[:-2]
        with st.expander(
            f"Previous questions ({sum(1 for item in older_messages if item.get('role') == 'user')})",
            expanded=False,
        ):
            for index, message in enumerate(older_messages):
                render_message(index, message)

    latest_start = max(0, len(messages) - 2)
    for index in range(latest_start, len(messages)):
        render_message(index, messages[index])


def _render_config_error(runtime: Runtime) -> None:
    st.error("The assistant is not configured correctly.")
    if runtime.error:
        st.markdown(f"**Details:** {runtime.error}")
    st.info(
        "On Streamlit Community Cloud, set `GROQ_API_KEY` and `LLM_MODEL_ID` in "
        "**App settings → Secrets**. Locally, put them in `.env` or "
        "`.streamlit/secrets.toml`."
    )
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
        "each stage reports its own progress. On Streamlit Community Cloud the "
        "filesystem is ephemeral, so a reboot may require re-initialization."
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
    get_suggested_questions.clear()
    st.rerun()


def _render_chat_input(runtime: Runtime) -> None:
    """Render an editable composer and process each explicit submission."""
    from src.services.chat import ChatUnavailableError, process_question

    with st.form("chat_form", clear_on_submit=False, border=False):
        input_col, send_col = st.columns(
            [10, 1],
            vertical_alignment="center",
        )
        with input_col:
            question = st.text_input(
                "Question",
                key="question_input",
                placeholder="Ask HDFC MF Facts a question...",
                label_visibility="collapsed",
                max_chars=2000,
            )
        with send_col:
            submitted = st.form_submit_button(
                "↑",
                type="primary",
                width="stretch",
                key="send_question_button",
            )

    if not submitted:
        return

    question = (question or "").strip()
    if not question:
        st.warning("Please enter a question.")
        return
    if len(question) > 2000:
        st.warning("Please keep questions under 2000 characters.")
        return

    # Always create a response envelope for a submitted question. This keeps
    # the turn visible in chat history even if a backend dependency fails.
    try:
        with st.spinner("Checking the official documents…"):
            if not get_knowledge_base():
                envelope = {
                    "type": "error",
                    "answer": (
                        "The knowledge base is unavailable right now. "
                        "Please try again shortly or initialize the knowledge base."
                    ),
                }
            else:
                model = get_model()
                result = process_question(
                    runtime.config,
                    question,
                    None,  # Scheme is inferred from the user's question.
                    model=model,
                )
                envelope = result.envelope

    except ChatUnavailableError:
        envelope = {
            "type": "error",
            "answer": (
                "The knowledge base is unavailable right now. "
                "Please try again shortly or initialize the knowledge base."
            ),
        }
    except ValueError:
        st.warning("Please enter a valid question.")
        return
    except Exception as exc:
        safe_log(
            logger,
            "chat_request_failed",
            error_type=type(exc).__name__,
        )
        envelope = {
            "type": "error",
            "answer": (
                "The assistant could not process that question right now. "
                "Please try again."
            ),
        }

    display_question = (
        "[Personal information hidden]"
        if envelope.get("type") == "pii"
        else question
    )

    st.session_state.messages.extend(
        [
            {"role": "user", "display": display_question},
            {"role": "assistant", "envelope": envelope},
        ]
    )

    # Clear the composer on the next script run, before the text_input widget
    # is created. This avoids form clear-on-submit state issues on later turns.
    st.session_state["_clear_question_input"] = True
    st.rerun()


def _inject_ui_styles() -> None:
    """Apply the visual system used by the HDFC MF Facts reference layout."""
    st.markdown(
        dedent(
            """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #ffffff;
        }
        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.98);
        }
        [data-testid="stMainBlockContainer"] {
            width: min(900px, calc(100vw - 2rem)) !important;
            max-width: 900px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 2.5rem !important;
            padding-bottom: 7.5rem !important;
        }
        [data-testid="stForm"] {
            position: sticky;
            bottom: 0;
            z-index: 100;
            background: rgba(255, 255, 255, 0.98);
            padding: 0.65rem 0 0.35rem;
            border-top: 1px solid #edf0ed;
        }
        [data-testid="stTextInput"] input {
            min-height: 48px;
            border-radius: 14px;
            padding-left: 16px;
        }
        [data-testid="stFormSubmitButton"] button {
            min-height: 48px;
            height: 48px;
            border-radius: 13px;
            background: #079c68;
            color: #ffffff;
            border: 1px solid #079c68;
        }
        [data-testid="stFormSubmitButton"] button:hover {
            background: #067d55;
            border-color: #067d55;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 64px;
            height: auto !important;
            white-space: normal !important;
            overflow-wrap: anywhere;
            word-break: break-word;
            text-align: left;
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            padding: 14px 16px;
            border: 1px solid #e1e6e2;
            border-radius: 13px;
            background: #ffffff;
            color: #17211c;
            font-size: 0.95rem;
            line-height: 1.45;
            box-shadow: none;
        }
        div[data-testid="stButton"] > button p {
            white-space: normal !important;
            overflow-wrap: anywhere;
            word-break: break-word;
            text-align: left;
            margin: 0;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: #079c68;
            color: #079c68;
            background: #fbfefc;
        }
        div[data-testid="stButton"] > button:focus {
            border-color: #079c68;
            box-shadow: 0 0 0 1px #079c68;
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 14px;
        }
        </style>
        """
        ),
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render a spacious brand header with a GitHub link."""
    brand, actions = st.columns([6.5, 1.8], vertical_alignment="center")

    with brand:
        st.markdown(
            dedent(
                """
                <div style="
                    display:flex;
                    align-items:center;
                    gap:12px;
                    min-height:80px;
                    padding:22px 0;
                ">
                    <div style="
                        background:#079c68;
                        color:#ffffff;
                        width:42px;
                        height:42px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        border-radius:10px;
                        font-weight:800;
                        font-size:20px;
                    ">H</div>
                    <div style="line-height:1.4;">
                        <div style="
                            font-weight:800;
                            letter-spacing:.3px;
                            color:#17211c;
                            font-size:16px;
                        ">HDFC MF FACTS</div>
                        <div style="
                            font-size:13px;
                            color:#7b8491;
                            margin-top:3px;
                        ">Mutual fund facts, made clear</div>
                    </div>
                </div>
            """
            ),
            unsafe_allow_html=True,
        )

    with actions:
        st.markdown(
            dedent(
                f"""
                <div style="
                    display:flex;
                    justify-content:flex-end;
                    align-items:center;
                    gap:14px;
                    white-space:nowrap;
                    min-height:80px;
                    padding:22px 0;
                    font-size:12px;
                ">
                    <span style="
                        background:#e5f7ed;
                        color:#087a4b;
                        padding:7px 12px;
                        border-radius:999px;
                        font-weight:700;
                    ">Facts-only</span>
                    <a href="{GITHUB_URL}"
                       target="_blank"
                       rel="noopener noreferrer"
                       style="color:#667085;text-decoration:none;">
                        GitHub ↗
                    </a>
                </div>
            """
            ),
            unsafe_allow_html=True,
        )

    st.divider()


def render_hero() -> None:
    """Render the centered introduction from the supplied UI reference."""
    st.markdown(
        dedent(
            """
        <div style="max-width:700px;margin:30px auto 24px;text-align:center;">
            <div style="color:#079c68;font-size:11px;font-weight:800;
                letter-spacing:2px;margin-bottom:14px;">
                YOUR MUTUAL FUND FACT DESK
            </div>
            <h1 style="font-size:42px;line-height:1.08;letter-spacing:-1.3px;
                color:#101828;margin:0 0 12px;font-weight:750;">
                Investing terms,<br>
                <span style="color:#079c68;">made clearer.</span>
            </h1>
            <p style="font-size:16px;color:#737b88;margin:0 0 18px;">
                Understand scheme documents and mutual-fund terms in plain language.
            </p>
            <div style="display:inline-block;padding:12px 16px;
                background:#f3f7f5;border:1px solid #e1eae4;border-radius:12px;
                font-size:13px;color:#344054;">
                Hi! Ask me facts about HDFC Mutual Fund schemes.
            </div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def render_examples() -> None:
    """Render factual example questions in a fully readable two-column grid."""
    st.markdown("### Try asking")
    st.caption(
        "Factual prompts with verified retrieval coverage."
    )
    questions = list(get_suggested_questions())[:8]
    for row_index in range(0, len(questions), 2):
        row = questions[row_index : row_index + 2]
        columns = st.columns(2, gap="medium")
        for column_index, question in enumerate(row):
            with columns[column_index]:
                if st.button(
                    question,
                    key=f"example_{row_index}_{column_index}",
                    width="stretch",
                    help=question,
                ):
                    _queue_followup(question)


def render_empty_answer_state() -> None:
    """Render the neutral answer placeholder before the first question."""
    st.markdown(
        dedent(
            """
        <div style="text-align:center;padding:28px 12px 16px;">
            <div style="width:34px;height:34px;margin:0 auto 8px;
                border:1px solid #e1eae4;border-radius:11px;
                display:flex;align-items:center;justify-content:center;">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                    stroke="#079c68" stroke-width="1.8" stroke-linecap="round"
                    stroke-linejoin="round" aria-hidden="true">
                    <path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7
                        8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8
                        8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5
                        a8.5 8.5 0 0 1 8 8v.5z"/>
                </svg>
            </div>
            <div style="font-weight:650;color:#344054;">
                Your answer will appear here
            </div>
            <div style="font-size:13px;color:#7b8491;margin-top:7px;">
                Choose an example above or type a question below.
            </div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render HDFC MF Facts with persistent examples and multi-turn chat."""
    _inject_ui_styles()
    render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    # Reset the submitted input before its widget is instantiated on this run.
    if st.session_state.get("_clear_question_input", False):
        st.session_state["question_input"] = ""
        st.session_state["_clear_question_input"] = False

    runtime = get_runtime()
    if runtime.config is None or runtime.error:
        if not st.session_state.messages:
            render_hero()
        _render_config_error(runtime)
        return

    if runtime.warnings:
        with st.expander("Deployment notes", expanded=True):
            for warning in runtime.warnings:
                st.warning(warning)
            st.caption(
                "Add the missing values in Streamlit Secrets (or local `.env`), "
                "then click **Reload configuration** below or reboot the app."
            )
            if st.button("Reload configuration", key="reload_after_warnings"):
                st.cache_resource.clear()
                st.rerun()

    if not artifacts_present(runtime.config):
        if not st.session_state.messages:
            render_hero()
        _render_setup(runtime.config)
        st.caption(DISCLAIMER)
        return

    if not st.session_state.messages:
        render_hero()

    # Keep suggested prompts above the answer/conversation area on every turn.
    render_examples()

    if st.session_state.messages:
        _render_history()
    else:
        render_empty_answer_state()

    st.caption(DISCLAIMER)
    _render_chat_input(runtime)


if __name__ == "__main__":
    main()
