from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.services.chat import ChatUnavailableError, process_question

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
URL = "https://files.hdfcfund.com/KIM/flexi-cap.pdf"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.llm.model = "test-model"
    cfg.safety.cache_answers = False
    return cfg


def _answer() -> dict:
    return {
        "type": "answer",
        "answer": f"The exit load is 1% for redemptions within one year. Source: {URL}",
        "source_url": URL,
        "source_name": "HDFC AMC",
        "as_on_date": "2026-01-02",
        "proof": "Exit load is 1% for redemptions within one year.",
        "can_do": [],
        "chunks_used": ["c1"],
        "request_id": "req123",
    }


def _fail(*args, **kwargs):
    raise AssertionError("pipeline should not run for this route")


def test_pii_is_refused_before_pipeline(config):
    result = process_question(
        config,
        "My PAN is ABCDE1234F and my account number is 123456789.",
        answer_fn=_fail,
        request_id="req-pii",
    )

    assert result.route == "pii"
    assert result.request_id == "req-pii"
    assert result.envelope["type"] == "pii"
    assert "ABCDE1234F" not in str(result.envelope)
    assert "123456789" not in str(result.envelope)


def test_advice_is_refused_before_pipeline(config):
    result = process_question(
        config,
        "Should I buy HDFC Mid Cap Fund?",
        answer_fn=_fail,
    )

    assert result.route == "refusal_advice"
    assert result.envelope["type"] == "refusal_advice"


def test_clarify_for_scheme_sensitive_question(config):
    result = process_question(
        config,
        "What is the exit load?",
        answer_fn=_fail,
    )

    assert result.route == "clarify"
    assert result.envelope["type"] == "clarify"
    assert len(result.envelope["can_do"]) == 4


def test_selected_scheme_is_passed_to_pipeline(config):
    calls = []

    def fake_answer(cfg, question, scheme=None, model=None):
        calls.append((question, scheme, model))
        return _answer()

    result = process_question(
        config,
        "What is the exit load?",
        "mid_cap",
        model=object(),
        answer_fn=fake_answer,
    )

    assert result.envelope["type"] == "answer"
    assert calls[0][1] == "mid_cap"
    assert calls[0][2] is not None


def test_multi_scheme_adds_followups(config):
    result = process_question(
        config,
        "What is the exit load of HDFC Flexi Cap Fund and HDFC Mid Cap Fund?",
        answer_fn=lambda cfg, question, scheme=None, model=None: _answer(),
    )

    assert result.envelope["type"] == "answer"
    assert result.envelope["can_do"] == [
        "What is the exit load of HDFC Mid Cap Fund?"
    ]


def test_unavailable_pipeline_raises_typed_error(config):
    def unavailable(*args, **kwargs):
        raise ValueError("collection unavailable")

    with pytest.raises(ChatUnavailableError):
        process_question(
            config,
            "What is the exit load of HDFC Flexi Cap Fund?",
            answer_fn=unavailable,
        )


def test_unexpected_error_is_safe(config):
    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    result = process_question(
        config,
        "What is the exit load of HDFC Flexi Cap Fund?",
        answer_fn=explode,
    )

    assert result.envelope["type"] == "error"
    assert "secret internal detail" not in str(result.envelope)


def test_empty_question_rejected(config):
    with pytest.raises(ValueError, match="must not be empty"):
        process_question(config, "   ", answer_fn=_fail)
