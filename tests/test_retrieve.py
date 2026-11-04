from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.rag import retrieve as retrieve_module
from src.rag.retrieve import RetrievedChunk, retrieve

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.retrieval.top_k = 5
    cfg.retrieval.score_threshold = 0.35
    cfg.retrieval.max_context_chunks = 3
    return cfg


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def query(self, **kwargs):
        return {
            "ids": [[row["chunk_id"] for row in self.rows]],
            "documents": [[row["text"] for row in self.rows]],
            "metadatas": [[row["metadata"] for row in self.rows]],
            "distances": [[row["distance"] for row in self.rows]],
        }


def _row(chunk_id: str, distance: float, text: str, **metadata):
    base = {
        "source_id": chunk_id,
        "url": "https://www.hdfcfund.com/fund.pdf",
        "publisher": "HDFC AMC",
        "doc_type": "kim",
        "scheme": "flexi_cap",
        "topic": "exit_load",
        "as_on_date": "2026-01-02",
        "section": "Exit load",
        "text_hash": chunk_id,
    }
    base.update(metadata)
    return {"chunk_id": chunk_id, "distance": distance, "text": text, "metadata": base}


def _install(monkeypatch, rows):
    monkeypatch.setattr(
        retrieve_module, "get_collection", lambda config: FakeCollection(rows)
    )


def test_retrieve_drops_below_threshold_and_caps_context(config, monkeypatch):
    _install(
        monkeypatch,
        [
            _row("a", 0.1, "Exit load is one percent."),
            _row("b", 0.2, "Exit load applies within one year."),
            _row("c", 0.3, "Exit load is nil after one year."),
            _row("d", 0.9, "Unrelated catalogue text."),
        ],
    )
    config.retrieval.max_context_chunks = 2

    result = retrieve(config, "What is the exit load?", model=FakeModel())

    assert [chunk.chunk_id for chunk in result.chunks] == ["a", "b"]
    assert result.debug["dropped"]["below_threshold"] == 1
    assert result.debug["dropped"]["over_context_limit"] == 1


def test_retrieve_dedupes_hash_and_near_identical_text(config, monkeypatch):
    _install(
        monkeypatch,
        [
            _row("a", 0.1, "Exit load is one percent for redemptions.", text_hash="same"),
            _row("b", 0.2, "Different words.", text_hash="same"),
            _row("c", 0.3, "Exit load is one percent for redemptions."),
            _row("d", 0.4, "A genuinely different passage about SIP."),
        ],
    )

    result = retrieve(config, "exit load", model=FakeModel())

    assert [chunk.chunk_id for chunk in result.chunks] == ["a", "d"]
    assert result.debug["dropped"]["duplicate"] == 2


def test_retrieve_prefers_specific_documents_within_window(config, monkeypatch):
    _install(
        monkeypatch,
        [
            _row("catalogue", 0.10, "Catalogue overview.", doc_type="catalogue"),
            _row("kim", 0.14, "KIM exit load details.", doc_type="kim"),
            _row("faq", 0.05, "FAQ overview.", doc_type="faq"),
        ],
    )

    result = retrieve(config, "exit load", model=FakeModel())

    assert [chunk.chunk_id for chunk in result.chunks] == ["faq", "kim", "catalogue"]


def test_retrieve_debug_excludes_raw_question(config, monkeypatch):
    _install(monkeypatch, [_row("a", 0.1, "Exit load is one percent.")])

    result = retrieve(
        config,
        "UniqueQuestionToken about exit load?",
        model=FakeModel(),
        persist_debug=True,
    )

    serialized = json.dumps(result.debug)
    assert "UniqueQuestionToken" not in serialized
    assert "question" not in result.debug

    trace = config.app.artifact_dir / "retrieval" / f"{result.debug['request_id']}.json"
    assert trace.exists()
    assert "UniqueQuestionToken" not in trace.read_text(encoding="utf-8")


def test_retrieved_chunk_accessors():
    chunk = RetrievedChunk(
        chunk_id="a",
        score=0.9,
        text="text",
        metadata={"url": "https://www.hdfcfund.com/x", "section": "Exit load"},
    )
    assert chunk.url == "https://www.hdfcfund.com/x"
    assert chunk.section == "Exit load"
    assert chunk.to_dict()["score"] == 0.9
