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
            _row(
                "catalogue",
                0.10,
                "Catalogue overview.",
                doc_type="catalogue",
                topic="general",
                section="Overview",
            ),
            _row(
                "kim",
                0.14,
                "KIM exit load details.",
                doc_type="kim",
                topic="exit_load",
                section="Exit load",
            ),
            _row(
                "faq",
                0.05,
                "FAQ overview.",
                doc_type="faq",
                topic="general",
                section="FAQ",
            ),
        ],
    )

    result = retrieve(config, "exit load", model=FakeModel())

    # Topic/lexical signal + official-doc preference should surface KIM first.
    assert [chunk.chunk_id for chunk in result.chunks] == ["kim", "faq", "catalogue"]


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


def test_retrieve_boosts_matching_topic_over_generic_definition(
    config, monkeypatch
):
    _install(
        monkeypatch,
        [
            _row(
                "defs",
                0.20,
                "A definition.",
                topic="definitions",
                section="Definitions",
            ),
            _row(
                "exit",
                0.30,
                "Exit load is 1% within one year.",
                topic="exit_load",
                section="Exit load",
            ),
        ],
    )

    result = retrieve(
        config, "What is the exit load of HDFC ELSS Tax Saver?", model=FakeModel()
    )

    assert result.debug["detected_topic"] == "exit_load"
    assert [chunk.chunk_id for chunk in result.chunks] == ["exit"]
    assert result.chunks[0].raw_score == pytest.approx(0.70)
    assert result.debug["candidates"][0]["chunk_id"] == "exit"
    assert result.debug["candidates"][0]["topic_match"] is True


def test_retrieve_boosts_topic_from_chunk_text_without_metadata(config, monkeypatch):
    _install(
        monkeypatch,
        [
            _row(
                "generic",
                0.15,
                "General scheme overview without the target fee.",
                topic="general",
                section="Overview",
            ),
            _row(
                "text_hit",
                0.28,
                "An Exit Load of 1.00% applies within one year.",
                topic="general",
                section="Fees",
            ),
        ],
    )

    result = retrieve(config, "What is the exit load?", model=FakeModel())

    assert result.debug["detected_topic"] == "exit_load"
    assert result.chunks[0].chunk_id == "text_hit"
    assert result.debug["candidates"][0]["topic_match"] is True


def test_retrieve_diversifies_selected_sources(config, monkeypatch):
    config.retrieval.max_context_chunks = 3
    _install(
        monkeypatch,
        [
            _row("a1", 0.10, "Exit load page detail one.", source_id="page"),
            _row("a2", 0.11, "Exit load page detail two.", source_id="page"),
            _row("b1", 0.12, "Exit load kim detail one.", source_id="kim"),
            _row("c1", 0.13, "Exit load sid detail one.", source_id="sid"),
        ],
    )

    result = retrieve(config, "What is the exit load?", model=FakeModel())

    selected_sources = [
        chunk.metadata.get("source_id") for chunk in result.chunks
    ]
    assert selected_sources == ["page", "kim", "sid"]


def test_retrieve_prefers_groww_risk_phrasing_over_empty_groww(config, monkeypatch):
    """Groww pages often say 'Very High Risk' without the word riskometer."""
    config.retrieval.max_context_chunks = 5
    _install(
        monkeypatch,
        [
            _row(
                "groww_noise",
                0.20,
                "Invest in stocks, ETFs, IPOs with fast orders. Track returns.",
                source_id="groww_elss_page",
                doc_type="groww_page",
                publisher="Groww",
                scheme="elss",
                topic="general",
                section="Nav",
                url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            ),
            _row(
                "hdfc_value",
                0.22,
                "Riskometer: Very High",
                source_id="hdfc_elss_page",
                doc_type="scheme_page",
                publisher="HDFC AMC",
                scheme="elss",
                topic="riskometer",
                section="Riskometer",
                url="https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct",
            ),
            _row(
                "groww_value",
                0.24,
                "HDFC ELSS Tax Saver Fund Direct Plan Growth is rated Very High risk.",
                source_id="groww_elss_page",
                doc_type="groww_page",
                publisher="Groww",
                scheme="elss",
                topic="riskometer",
                section="Fund details",
                url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            ),
        ],
    )

    result = retrieve(
        config,
        "What is the riskometer of HDFC ELSS Tax Saver?",
        scheme="elss",
        model=FakeModel(),
    )

    assert result.chunks[0].chunk_id == "groww_value"
    assert "groww_noise" not in {chunk.chunk_id for chunk in result.chunks}


def test_retrieve_prefers_value_bearing_riskometer_chunk(config, monkeypatch):
    config.retrieval.max_context_chunks = 5
    _install(
        monkeypatch,
        [
            _row(
                "mention",
                0.10,
                "See the scheme Riskometer and Benchmark Riskometer notes.",
                source_id="hdfc_large_cap_kim",
                doc_type="kim",
                scheme="large_cap",
                topic="riskometer",
                section="Riskometer",
            ),
            _row(
                "value",
                0.22,
                "Riskometer: Very High",
                source_id="hdfc_large_cap_page",
                doc_type="scheme_page",
                scheme="large_cap",
                topic="riskometer",
                section="Riskometer",
            ),
            _row(
                "noise",
                0.12,
                "https://www.amfiindia.com/online-center/risk-o-meter",
                source_id="hdfc_large_cap_kim",
                doc_type="kim",
                scheme="large_cap",
                topic="general",
                section="Links",
            ),
        ],
    )

    result = retrieve(
        config,
        "What is the riskometer of HDFC Large Cap Fund?",
        scheme="large_cap",
        model=FakeModel(),
    )

    assert result.chunks[0].chunk_id == "value"
    assert all("Very High" in chunk.text or "Riskometer" in chunk.text for chunk in result.chunks)
    assert len(result.chunks) <= 3


def test_retrieve_prefers_groww_page_for_scheme_facts(config, monkeypatch):
    config.retrieval.max_context_chunks = 3
    _install(
        monkeypatch,
        [
            _row(
                "noise",
                0.20,
                "General mutual fund education about statements and folios.",
                source_id="groww_hdfc_amc",
                doc_type="groww_page",
                publisher="Groww",
                scheme="amc_wide",
                topic="general",
                section="Overview",
                url="https://groww.in/mutual-funds/amc/hdfc-mutual-funds",
            ),
            _row(
                "value",
                0.25,
                "Riskometer: Very High",
                source_id="groww_large_cap_page",
                doc_type="groww_page",
                publisher="Groww",
                scheme="large_cap",
                topic="riskometer",
                section="Scheme Facts",
                url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            ),
            _row(
                "other",
                0.18,
                "Account statement errors may be a sign of fraud.",
                source_id="other_page",
                doc_type="education",
                scheme="amc_wide",
                topic="statements",
                section="Education",
            ),
        ],
    )

    result = retrieve(
        config,
        "What is the riskometer of HDFC Large Cap Fund?",
        scheme="large_cap",
        model=FakeModel(),
    )

    assert result.chunks[0].chunk_id == "value"


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
