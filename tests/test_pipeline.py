from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.rag import pipeline as pipeline_module
from src.rag.generate import GenerationError
from src.rag.pipeline import answer_question
from src.rag.retrieve import RetrievalResult, RetrievedChunk

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
URL = "https://files.hdfcfund.com/KIM/flexi-cap.pdf"
TEXT = "Exit load is 1% for redemptions within one year."


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.llm.model = "test-model"
    cfg.safety.cache_answers = True
    cfg.safety.max_repair_retries = 1
    return cfg


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        score=0.9,
        text=TEXT,
        metadata={
            "url": URL,
            "publisher": "HDFC AMC",
            "doc_type": "kim",
            "scheme": "flexi_cap",
            "section": "Exit load",
            "as_on_date": "2026-01-02",
        },
    )


def _retrieval(chunks=None) -> RetrievalResult:
    return RetrievalResult(
        chunks=[_chunk()] if chunks is None else chunks,
        debug={"request_id": "req123"},
    )


def _candidate(answer: str | None = None) -> str:
    return json.dumps(
        {
            "answer": answer or f"The exit load is 1%. Source: {URL}",
            "source_url": URL,
            "proof": "Exit load is 1%",
        }
    )


def test_answer_question_builds_envelope_and_caches(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())

    envelope = answer_question(
        config,
        "UniqueQuestionToken for exit load",
        complete_fn=lambda cfg, messages: _candidate(),
    )

    assert envelope["type"] == "answer"
    assert envelope["source_url"] == URL
    assert envelope["source_name"] == "HDFC AMC"
    assert envelope["as_on_date"] == "2026-01-02"
    assert envelope["chunks_used"] == ["c1"]
    assert envelope["request_id"] == "req123"

    files = list((config.app.artifact_dir / "cache" / "answers").glob("*.json"))
    assert len(files) == 1
    cached_text = files[0].read_text(encoding="utf-8")
    assert "UniqueQuestionToken" not in cached_text


def test_cache_hit_skips_retrieval(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())
    first = answer_question(
        config, "Same cached question", complete_fn=lambda cfg, messages: _candidate()
    )

    def fail(*args, **kwargs):
        raise AssertionError("retrieve should not run on a cache hit")

    monkeypatch.setattr(pipeline_module, "retrieve", fail)
    second = answer_question(
        config, "Same cached question", complete_fn=lambda cfg, messages: _candidate()
    )

    assert second == first


def test_validator_failure_is_repaired_once(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())
    responses = iter(
        [
            _candidate(answer=f"One. Two. Three. Four. Source: {URL}"),
            _candidate(),
        ]
    )
    calls = []

    def complete_fn(cfg, messages):
        calls.append(messages)
        return next(responses)

    envelope = answer_question(config, "exit load", complete_fn=complete_fn)

    assert envelope["type"] == "answer"
    assert len(calls) == 2
    assert "failed validation" in "\n".join(m["content"] for m in calls[1])


def test_failed_repair_returns_not_in_sources(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())
    responses = iter(
        [
            _candidate(answer=f"One. Two. Three. Four. Source: {URL}"),
            _candidate(answer=f"One. Two. Three. Four. Source: {URL}"),
        ]
    )
    envelope = answer_question(
        config, "exit load", complete_fn=lambda cfg, messages: next(responses)
    )

    assert envelope["type"] == "not_in_sources"
    assert envelope["source_url"] == pipeline_module.SCHEME_DOCUMENTS_URL


def test_no_chunks_returns_not_in_sources(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval([]))

    envelope = answer_question(
        config, "exit load", complete_fn=lambda cfg, messages: _candidate()
    )

    assert envelope["type"] == "not_in_sources"
    assert envelope["chunks_used"] == []


def test_topic_mismatch_still_generates_when_chunk_text_is_usable(
    config, monkeypatch
):
    """Metadata topic mismatch must not block generation if chunk text is usable."""
    chunk = _chunk()
    chunk.metadata["topic"] = "definitions"
    retrieval = RetrievalResult(
        chunks=[chunk],
        debug={"request_id": "req123", "detected_topic": "exit_load"},
    )
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: retrieval)

    envelope = answer_question(
        config,
        "What is the exit load of HDFC Flexi Cap Fund?",
        complete_fn=lambda cfg, messages: _candidate(),
    )

    assert envelope["type"] == "answer"
    assert envelope["chunks_used"] == ["c1"]
    assert retrieval.debug["evidence_assessment"]["retrieved_chunks"] == 1
    assert retrieval.debug["evidence_assessment"]["topic_matches"] == 0


def test_model_says_not_in_sources(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())

    envelope = answer_question(
        config,
        "exit load",
        complete_fn=lambda cfg, messages: '{"not_in_sources": true}',
    )

    assert envelope["type"] == "not_in_sources"


def test_generation_error_returns_error(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())

    def fail(cfg, messages):
        raise GenerationError("provider down")

    envelope = answer_question(config, "exit load", complete_fn=fail)

    assert envelope["type"] == "error"
    assert envelope["source_url"] is None


def test_empty_question_rejected(config):
    with pytest.raises(ValueError, match="must not be empty"):
        answer_question(config, "   ")


def test_provider_failure_retries_once_then_succeeds(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())
    calls = []

    def complete_fn(cfg, messages):
        calls.append(messages)
        if len(calls) == 1:
            raise GenerationError("provider timeout")
        return _candidate()

    envelope = answer_question(config, "retry provider", complete_fn=complete_fn)

    assert envelope["type"] == "answer"
    assert len(calls) == 2


def test_multi_scheme_focuses_first_scheme_for_generation(config, monkeypatch):
    seen = {}

    def fake_retrieve(cfg, question, scheme=None, **kwargs):
        seen["scheme"] = scheme
        return _retrieval()

    monkeypatch.setattr(pipeline_module, "retrieve", fake_retrieve)

    def complete_fn(cfg, messages):
        seen["prompt"] = messages[-1]["content"]
        return _candidate()

    envelope = answer_question(
        config,
        "What is the exit load of HDFC Flexi Cap Fund and HDFC Mid Cap Fund?",
        model=object(),
        complete_fn=complete_fn,
    )

    assert envelope["type"] == "answer"
    assert seen["scheme"] == "flexi_cap"
    assert "HDFC Mid Cap Fund" not in seen["prompt"]


def test_pii_echo_is_replaced_with_pii_response(config, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: _retrieval())
    bad = json.dumps(
        {
            "answer": f"My PAN is ABCDE1234F. Source: {URL}",
            "source_url": URL,
            "proof": "Exit load is 1%",
        }
    )

    envelope = answer_question(config, "pii in output", complete_fn=lambda cfg, m: bad)

    assert envelope["type"] == "pii"
    assert "ABCDE1234F" not in json.dumps(envelope)


def test_missing_source_date_falls_back_to_retrieved_at(config, monkeypatch):
    chunk = _chunk()
    chunk.metadata.pop("as_on_date")
    chunk.metadata["fetched_at"] = "2026-01-05T07:00:00+00:00"
    monkeypatch.setattr(
        pipeline_module,
        "retrieve",
        lambda *a, **k: RetrievalResult(chunks=[chunk], debug={"request_id": "req123"}),
    )

    envelope = answer_question(
        config, "missing date", complete_fn=lambda cfg, m: _candidate()
    )

    assert envelope["as_on_date"] is None
    assert envelope["retrieved_at"].startswith("2026-01-05")
