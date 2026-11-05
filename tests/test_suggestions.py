from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.rag import suggestions as suggestions_module
from src.rag.retrieve import RetrievalResult, RetrievedChunk
from src.rag.suggestions import (
    build_suggestions,
    load_suggestions,
    resolve_suggestions,
    write_suggestions,
)
from src.rag.topics import detect_topic
from src.suggestions import DEFAULT_SUGGESTIONS

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    return cfg


def _retrieval(question: str, scheme: str | None, score: float = 0.9):
    metadata = {
        "topic": detect_topic(question),
        "scheme": scheme or "amc_wide",
        "source_id": "source",
    }
    return RetrievalResult(
        chunks=[RetrievedChunk("chunk", score, question, metadata)],
        debug={"detected_topic": metadata["topic"]},
    )


def test_build_suggestions_keeps_only_verified_questions(config, monkeypatch):
    monkeypatch.setattr(
        suggestions_module,
        "retrieve",
        lambda cfg, question, scheme=None, **kwargs: _retrieval(question, scheme),
    )

    items = build_suggestions(config, model=object(), limit=6)

    assert [item["question"] for item in items] == list(DEFAULT_SUGGESTIONS)
    assert all(item["score"] == 0.9 for item in items)


def test_build_suggestions_drops_low_score(config, monkeypatch):
    monkeypatch.setattr(
        suggestions_module,
        "retrieve",
        lambda cfg, question, scheme=None, **kwargs: _retrieval(
            question, scheme, score=0.1
        ),
    )

    assert build_suggestions(config, model=object(), limit=6) == []


def test_build_suggestions_requires_matching_topic(config, monkeypatch):
    monkeypatch.setattr(
        suggestions_module,
        "retrieve",
        lambda *args, **kwargs: RetrievalResult(
            chunks=[
                RetrievedChunk(
                    "chunk",
                    0.9,
                    "definition",
                    {"topic": "capital_gains", "scheme": "flexi_cap"},
                )
            ],
            debug={},
        ),
    )

    assert build_suggestions(config, model=object(), limit=1) == []


def test_suggestion_artifact_roundtrip(config, monkeypatch):
    items = [
        {
            "question": DEFAULT_SUGGESTIONS[0],
            "topic": "exit_load",
            "scheme": "flexi_cap",
            "score": 0.9,
            "chunk_id": "chunk",
            "source_id": "source",
        }
    ]
    monkeypatch.setattr(suggestions_module, "build_suggestions", lambda *a, **k: items)

    result = write_suggestions(config, model=object(), limit=6)
    cached = load_suggestions(config, limit=6)

    assert result["reused"] is False
    assert cached == items
    assert resolve_suggestions(config, limit=6) == [DEFAULT_SUGGESTIONS[0]]


def test_resolve_suggestions_falls_back_to_curated_defaults(config):
    assert resolve_suggestions(config, limit=6) == list(DEFAULT_SUGGESTIONS)
