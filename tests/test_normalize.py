from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.ingest.inventory import SourceSpec
from src.ingest.loader import FetchResult, build_manifest, write_manifest
from src.ingest.normalize import (
    TextRecord,
    dedupe_records,
    extract_date,
    normalize_html,
    normalize_source,
    records_from_pdf_pages,
    run_normalize,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

HTML = """
<html>
  <head><style>.x { color: red; }</style></head>
  <body>
    <header><nav>Menu Home About</nav></header>
    <div class="cookie-banner">Accept all cookies to continue</div>
    <h1>HDFC Mid Cap Fund</h1>
    <p>This page summarizes the scheme information and key facts for investors.</p>
    <h2>Exit Load</h2>
    <p>Exit load of 1.00% applies if units are redeemed within 1 year.</p>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Expense ratio</td><td>1.31%</td></tr>
    </table>
    <h2>Exit Load</h2>
    <p>Exit load of 1.00% applies if units are redeemed within 1 year.</p>
    <script>var ignored = "do not include";</script>
    <p>Dated November 21, 2025</p>
  </body>
</html>
""".encode("utf-8")


@pytest.fixture
def spec():
    return SourceSpec(
        source_id="hdfc_mid_cap_page",
        url="https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/regular",
        publisher="HDFC AMC",
        doc_type="scheme_page",
        scheme="mid_cap",
        title="HDFC Mid Cap Fund scheme page",
        expected_topics=["expense_ratio", "exit_load"],
    )


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.vector_store.path = tmp_path / "chroma"
    return cfg


def test_extract_date_formats():
    assert extract_date("as on 2026-10-18") == "2026-10-18"
    assert extract_date("dated November 21, 2025") == "2025-11-21"
    assert extract_date("21 Nov 2025") == "2025-11-21"
    assert extract_date("31/02/2025") is None
    assert extract_date("no date here") is None


def test_normalize_html_extracts_sections_and_table(spec):
    records = normalize_html(HTML, spec, "text/html; charset=utf-8", "2026-11-01T00:00:00+00:00")
    assert records
    assert all(record.source_id == spec.source_id for record in records)
    assert all(record.publisher == "HDFC AMC" for record in records)
    assert all(record.as_on_date == "2025-11-21" for record in records)

    exit_load = [record for record in records if "Exit load of 1.00%" in record.text]
    assert len(exit_load) == 1
    assert exit_load[0].section_path == ["HDFC Mid Cap Fund", "Exit Load"]

    tables = [record for record in records if "| Parameter | Value |" in record.text]
    assert len(tables) == 1
    assert "| Expense ratio | 1.31% |" in tables[0].text

    combined = " ".join(record.text for record in records)
    assert "Menu Home" not in combined
    assert "Accept all cookies" not in combined
    assert "do not include" not in combined


def test_normalize_html_extracts_scheme_fact_cards(spec):
    html = b"""
    <html><body>
      <div class="fact">
        <p class="title">TER</p>
        <p class="description">1.31</p>
      </div>
      <div class="fact">
        <div class="title">Min SIP</div>
        <p class="description">\xe2\x82\xb9 100</p>
      </div>
      <div class="fact">
        <p class="title">Riskometer</p>
        <p class="description">Very High</p>
      </div>
      <div class="fact">
        <p class="title">Entry Load</p>
        <p class="description">NA</p>
      </div>
    </body></html>
    """
    records = normalize_html(html, spec, "text/html", None)
    texts = [record.text for record in records]
    assert "Total Expense Ratio (TER): 1.31%" in texts
    assert "Minimum SIP: \u20b9 100" in texts
    assert "Riskometer: Very High" in texts
    assert "Entry Load: NA" not in texts
    assert all(record.section_path[:1] == ["Scheme Facts"] for record in records)
    assert ["Scheme Facts", "Total Expense Ratio (TER)"] in [
        record.section_path for record in records
    ]


def test_heading_levels_reset_sections(spec):
    html = b"<h1>Fund</h1><h2>A</h2><p>First paragraph with enough characters.</p><h3>A1</h3><p>Second paragraph with enough characters.</p>"
    records = normalize_html(html, spec, "text/html", None)
    assert records[0].section_path == ["Fund", "A"]
    assert records[1].section_path == ["Fund", "A", "A1"]


def test_pdf_records_from_pages(spec):
    pages = [
        "HDFC ELSS Fund\nExit Load\nThe exit load is nil for this scheme.\n\nMinimum SIP\nMinimum SIP amount is 500 rupees.\n",
        "Lock-in\nThere is a lock-in period of three years.\n",
    ]
    records = records_from_pdf_pages(pages, spec, "2025-11-21", None)
    texts = [record.text for record in records]
    assert any("The exit load is nil" in text for text in texts)
    assert any("Minimum SIP amount is 500" in text for text in texts)
    assert any("lock-in period of three years" in text for text in texts)
    assert all(record.page in {1, 2} for record in records)
    assert any(record.section_path == ["Lock-in"] for record in records)


def test_dedupe_records():
    base = TextRecord("s", "u", "HDFC AMC", "kim", "elss", "t", None, ["A"], 1, "same text")
    duplicate = TextRecord("s", "u", "HDFC AMC", "kim", "elss", "t", None, ["A"], 1, "same text")
    other = TextRecord("s", "u", "HDFC AMC", "kim", "elss", "t", None, ["B"], 1, "same text")
    assert [record.section_path for record in dedupe_records([base, duplicate, other])] == [
        ["A"],
        ["B"],
    ]


def test_normalize_source_from_raw_html(config, spec):
    raw = config.app.artifact_dir / "raw" / "hdfc_mid_cap_page.html"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(HTML)
    result = FetchResult(
        source_id=spec.source_id,
        url=spec.url,
        status="ok",
        content_type="text/html; charset=utf-8",
        fetched_at="2026-11-01T00:00:00+00:00",
        local_path="artifacts/raw/hdfc_mid_cap_page.html",
    )
    records, pages, error = normalize_source(spec, result, config.project_root)
    assert error is None
    assert pages is None
    assert records


def test_run_normalize_writes_jsonl_and_coverage(config, spec):
    raw = config.app.artifact_dir / "raw" / "hdfc_mid_cap_page.html"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(HTML)
    result = FetchResult(
        source_id=spec.source_id,
        url=spec.url,
        status="ok",
        content_type="text/html; charset=utf-8",
        fetched_at="2026-11-01T00:00:00+00:00",
        local_path="artifacts/raw/hdfc_mid_cap_page.html",
    )
    write_manifest(build_manifest([result], config), config)

    inventory = config.project_root / "sources.csv"
    inventory.write_text(
        "source_id,url,publisher,doc_type,scheme,title,expected_topics\n"
        f"{spec.source_id},{spec.url},HDFC AMC,scheme_page,mid_cap,Test,expense_ratio|exit_load\n",
        encoding="utf-8",
    )

    payload = run_normalize(config, inventory_path=inventory)
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["records"] > 0

    jsonl = config.app.artifact_dir / "normalized" / "hdfc_mid_cap_page.jsonl"
    assert jsonl.exists()
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows and rows[0]["section_path"]

    coverage = json.loads(
        (config.app.artifact_dir / "normalized" / "_coverage.json").read_text(encoding="utf-8")
    )
    assert coverage["summary"]["ok"] == 1
    assert coverage["sources"][0]["as_on_date"] == "2025-11-21"


def test_run_normalize_requires_manifest(config):
    with pytest.raises(ValueError, match="Run `--stage load` first"):
        run_normalize(config, inventory_path=CONFIG_PATH.parent / "sources.csv")


def test_run_normalize_marks_empty_page_unusable(config, spec):
    raw = config.app.artifact_dir / "raw" / "hdfc_mid_cap_page.html"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"<html><body><script>var x = 1;</script></body></html>")
    result = FetchResult(
        source_id=spec.source_id,
        url=spec.url,
        status="ok",
        content_type="text/html",
        fetched_at="2026-11-01T00:00:00+00:00",
        local_path="artifacts/raw/hdfc_mid_cap_page.html",
    )
    write_manifest(build_manifest([result], config), config)
    inventory = config.project_root / "sources.csv"
    inventory.write_text(
        "source_id,url,publisher,doc_type,scheme,title,expected_topics\n"
        f"{spec.source_id},{spec.url},HDFC AMC,scheme_page,mid_cap,Test,exit_load\n",
        encoding="utf-8",
    )
    payload = run_normalize(config, inventory_path=inventory)
    assert payload["summary"]["unusable"] == 1
    assert payload["sources"][0]["status"] == "unusable"
