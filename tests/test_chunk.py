from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.ingest.chunk import detect_topic, estimate_tokens, run_chunk

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.vector_store.path = tmp_path / "chroma"
    return cfg


def _record(
    text: str,
    section: str = "Fund Facts",
    page: int | None = None,
    section_path: list[str] | None = None,
) -> dict:
    return {
        "source_id": "hdfc_mid_cap_kim",
        "url": "https://www.hdfcfund.com/mid-cap-kim.pdf",
        "publisher": "HDFC AMC",
        "doc_type": "kim",
        "scheme": "mid_cap",
        "title": "KIM - HDFC Mid Cap Fund",
        "as_on_date": "2025-09-30",
        "section_path": section_path or [section],
        "page": page,
        "text": text,
        "fetched_at": "2026-11-01T00:00:00+00:00",
    }


def _write_normalized(config, records: list[dict], source_id: str = "hdfc_mid_cap_kim") -> Path:
    path = config.app.artifact_dir / "normalized" / f"{source_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            row = dict(record)
            row["source_id"] = source_id
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 401) == 101


def test_detect_topic_prefers_section_and_keywords():
    assert detect_topic("Exit Load", "A fee applies.") == "exit_load"
    assert detect_topic("Scheme Facts", "Total Expense Ratio (TER): 1.31%") == "expense_ratio"
    assert detect_topic("Scheme Facts", "Lock-in: 3 years") == "elss_lockin"
    assert detect_topic("Scheme Facts", "Minimum SIP: ₹ 100") == "min_sip"
    assert detect_topic("Scheme Facts", "Riskometer: Very High") == "riskometer"
    assert detect_topic("Performance", "Benchmark: Nifty 100 TRI") == "benchmark"
    assert detect_topic("Help", "How to download an account statement") == "statements"
    assert detect_topic("Education", "NAV stands for Net Asset Value.") == "definitions"
    assert detect_topic("About", "The scheme invests in equity.") == "general"


def test_run_chunk_writes_artifacts_and_covers_topics(config):
    records = [
        _record(
            "Total Expense Ratio (TER): 1.31%",
            section_path=["Scheme Facts", "Total Expense Ratio (TER)"],
        ),
        _record("Exit Load: 1.00% within 1 year, nil after 1 year.", "Exit Load"),
        _record("Minimum SIP: ₹ 100", section_path=["Scheme Facts", "Minimum SIP"]),
        _record("Lock-in: 3 years for ELSS investments.", section_path=["Scheme Facts", "Lock-in"]),
        _record("Riskometer: Very High", section_path=["Scheme Facts", "Riskometer"]),
        _record("Benchmark: Nifty Midcap 150 TRI", "Benchmark"),
        _record("You can download an account statement from the investor portal.", "Help"),
        _record("NAV stands for Net Asset Value per unit.", "Education"),
        _record("HDFC Mid Cap Fund is an open ended equity scheme.", "About"),
    ]
    _write_normalized(config, records)

    result = run_chunk(config)

    assert result["summary"]["chunks"] > 0
    assert result["summary"]["oversize"] == 0
    assert result["config"]["effective_target_tokens"] == 224
    assert result["config"]["max_seq_tokens"] == 256
    assert result["config"]["context_header_reserve_tokens"] == 32
    for topic in (
        "expense_ratio",
        "exit_load",
        "min_sip",
        "elss_lockin",
        "riskometer",
        "benchmark",
        "statements",
        "definitions",
    ):
        assert result["topic_coverage"][topic]["chunks"] > 0

    chunks_path = config.app.artifact_dir / "chunks" / "chunks.jsonl"
    plan_path = config.app.artifact_dir / "chunks" / "chunking_plan.json"
    stats_path = config.app.artifact_dir / "chunks" / "chunk_stats.json"
    assert chunks_path.exists() and plan_path.exists() and stats_path.exists()

    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    assert all(chunk["chunk_id"] and chunk["text"] for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["token_count"] <= 256 for chunk in chunks)
    assert all(chunk["topic"] for chunk in chunks)


def test_chunk_ids_are_stable(config):
    _write_normalized(config, [_record("Stable text for a stable chunk identifier.")])
    first = run_chunk(config)
    first_ids = [
        json.loads(line)["chunk_id"]
        for line in (config.app.artifact_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    second = run_chunk(config)
    second_ids = [
        json.loads(line)["chunk_id"]
        for line in (config.app.artifact_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert first["summary"]["chunks"] == second["summary"]["chunks"]
    assert first_ids == second_ids


def test_tables_are_kept_intact(config):
    table = "| Parameter | Value |\n| --- | --- |\n| Expense ratio | 1.31% |\n| Exit load | 1.00% |"
    _write_normalized(config, [_record(table, "Key Facts"), _record("Some prose about the scheme.", "About")])

    result = run_chunk(config)
    assert result["strategy_counts"] == {"table_preserving": 1}

    chunks = [
        json.loads(line)
        for line in (config.app.artifact_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(chunk["text"] == table for chunk in chunks)


def test_long_text_is_split_under_embedding_limit(config):
    long_text = " ".join(
        f"This is sentence number {index} with enough words to keep going." for index in range(200)
    )
    _write_normalized(config, [_record(long_text, "Long Section")])

    result = run_chunk(config)
    assert result["summary"]["chunks"] > 1
    assert result["summary"]["oversize"] == 0

    chunks = [
        json.loads(line)
        for line in (config.app.artifact_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(chunk["token_count"] <= 256 for chunk in chunks)
    assert all(len(chunk["text"]) <= 960 for chunk in chunks)


def test_run_chunk_requires_normalized_directory(config):
    with pytest.raises(ValueError, match="Run `--stage normalize` first"):
        run_chunk(config)
