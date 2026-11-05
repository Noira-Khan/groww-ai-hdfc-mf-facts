from __future__ import annotations

import dataclasses
import http.server
import threading
from pathlib import Path

import pytest

from src.config import load_config
from src.ingest.inventory import SourceSpec, load_inventory, validate_inventory
from src.ingest.loader import (
    build_manifest,
    fetch_all,
    fetch_source,
    merge_with_existing,
    read_manifest,
    run_load,
    write_manifest,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.vector_store.path = tmp_path / "chroma"
    cfg.ingestion.request_delay_ms = 0
    cfg.ingestion.max_retries = 2
    cfg.ingestion.request_timeout_s = 5
    return cfg


@pytest.fixture
def server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ok.html":
                body = b"<html><body><h1>Exit Load</h1><p>Exit load is 1%.</p></body></html>"
                self._send(200, "text/html; charset=utf-8", body)
            elif self.path == "/ok.pdf":
                body = b"%PDF-1.4\n%fake pdf for tests\n%%EOF\n"
                self._send(200, "application/pdf", body)
            elif self.path == "/mislabeled.pdf":
                body = b"%PDF-1.7\n%magic wins\n"
                self._send(200, "text/html", body)
            elif self.path == "/missing":
                self._send(404, "text/html", b"not found")
            elif self.path == "/server-error":
                self._send(500, "text/html", b"boom")
            else:
                self._send(404, "text/html", b"not found")

        def _send(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def spec(source_id: str, url: str, **overrides) -> SourceSpec:
    base = SourceSpec(
        source_id=source_id,
        url=url,
        publisher="Groww",
        doc_type="groww_page",
        scheme="large_cap",
        title="Test source",
        expected_topics=["exit_load"],
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def on_disk(result, config) -> Path:
    path = Path(result.local_path)
    return path if path.is_absolute() else config.project_root / path


def write_inventory(path: Path, rows: list[dict]) -> Path:
    columns = ["source_id", "url", "publisher", "doc_type", "scheme", "title", "expected_topics"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_inventory_parses_topics(tmp_path):
    path = write_inventory(
        tmp_path / "sources.csv",
        [
            {
                "source_id": "hdfc_test",
                "url": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/regular",
                "publisher": "HDFC AMC",
                "doc_type": "scheme_page",
                "scheme": "large_cap",
                "title": "Test",
                "expected_topics": "exit_load|min_sip",
            }
        ],
    )
    specs = load_inventory(path)
    assert len(specs) == 1
    assert specs[0].expected_topics == ["exit_load", "min_sip"]


def test_load_inventory_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("source_id,url\nx,https://example.com\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_inventory(path)


def test_shipped_inventory_is_valid(config):
    specs = load_inventory(CONFIG_PATH.parent / "sources.csv")
    validate_inventory(specs, config.ingestion.allowlist_publishers)
    assert 15 <= len(specs) <= 25
    assert len(specs) == 15
    assert all(spec.publisher == "Groww" for spec in specs)
    assert all(spec.doc_type == "groww_page" for spec in specs)
    assert all("groww.in" in spec.url for spec in specs)


def test_validate_rejects_host_mismatch(config):
    bad = spec(
        "groww_bad",
        "https://www.sebi.gov.in/legal/circulars/x.html",
    )
    with pytest.raises(ValueError, match="does not belong"):
        validate_inventory([bad], config.ingestion.allowlist_publishers)


def test_validate_rejects_unknown_publisher(config):
    bad = spec("blog_bad", "https://example.com/x", publisher="Random Blog")
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_inventory([bad], config.ingestion.allowlist_publishers)


def test_validate_rejects_non_http_url(config):
    bad = spec("groww_bad", "ftp://groww.in/mutual-funds/x")
    with pytest.raises(ValueError, match="absolute http"):
        validate_inventory([bad], config.ingestion.allowlist_publishers)


def test_validate_rejects_duplicate_ids(config):
    good = spec(
        "groww_dup",
        "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_inventory([good, good], config.ingestion.allowlist_publishers)


def test_validate_rejects_unknown_topic(config):
    bad = spec(
        "groww_bad",
        "https://groww.in/mutual-funds/x",
        expected_topics=["tax_advice"],
    )
    with pytest.raises(ValueError, match="unknown topics"):
        validate_inventory([bad], config.ingestion.allowlist_publishers)


def test_fetch_html_saves_file_with_checksum(config, server):
    result = fetch_source(spec("test_html", f"{server}/ok.html"), config, sleep_fn=lambda _: None)
    assert result.status == "ok"
    assert result.http_status == 200
    assert result.sha256 and len(result.sha256) == 64
    assert result.local_path.endswith("test_html.html")
    assert on_disk(result, config).read_bytes().startswith(b"<html>")
    assert result.bytes == on_disk(result, config).stat().st_size


def test_fetch_pdf_detected_by_magic_bytes(config, server):
    result = fetch_source(
        spec("test_pdf", f"{server}/mislabeled.pdf"), config, sleep_fn=lambda _: None
    )
    assert result.status == "ok"
    assert result.local_path.endswith(".pdf")


def test_fetch_missing_returns_failed(config, server):
    result = fetch_source(spec("test_missing", f"{server}/missing"), config, sleep_fn=lambda _: None)
    assert result.status == "failed"
    assert result.http_status == 404
    assert result.local_path is None


def test_fetch_retries_then_fails_on_server_error(config, server):
    result = fetch_source(
        spec("test_500", f"{server}/server-error"), config, sleep_fn=lambda _: None
    )
    assert result.status == "failed"
    assert "500" in (result.error or "")


def test_refetch_produces_same_sha(config, server):
    first = fetch_source(spec("test_same", f"{server}/ok.html"), config, sleep_fn=lambda _: None)
    second = fetch_source(spec("test_same", f"{server}/ok.html"), config, sleep_fn=lambda _: None)
    assert first.sha256 == second.sha256


def test_fetch_all_continues_and_manifest_written(config, server):
    specs = [
        spec("test_ok", f"{server}/ok.html"),
        spec("test_bad", f"{server}/missing"),
    ]
    results = fetch_all(specs, config, sleep_fn=lambda _: None)
    assert [item.status for item in results] == ["ok", "failed"]

    manifest = build_manifest(results, config)
    assert manifest["summary"] == {
        "total": 2,
        "ok": 1,
        "failed": 1,
        "bytes": on_disk(results[0], config).stat().st_size,
    }

    path = write_manifest(manifest, config)
    assert path.name == "manifest.json"
    assert read_manifest(config)["summary"]["ok"] == 1


def test_fetch_all_fail_fast(config, server):
    specs = [
        spec("test_bad", f"{server}/missing"),
        spec("test_ok", f"{server}/ok.html"),
    ]
    results = fetch_all(specs, config, fail_fast=True, sleep_fn=lambda _: None)
    assert len(results) == 1
    assert results[0].status == "failed"


def test_fetch_all_reports_progress(config, server):
    events = []
    specs = [
        spec("test_first", f"{server}/ok.html"),
        spec("test_second", f"{server}/missing"),
    ]

    fetch_all(
        specs,
        config,
        sleep_fn=lambda _: None,
        on_progress=lambda completed, total, result: events.append(
            (completed, total, result.source_id, result.status)
        ),
    )

    assert events == [
        (1, 2, "test_first", "ok"),
        (2, 2, "test_second", "failed"),
    ]


def test_progress_callback_errors_do_not_break_fetch(config, server):
    def broken(*args):
        raise RuntimeError("ui callback failed")

    results = fetch_all(
        [spec("test_ok", f"{server}/ok.html")],
        config,
        sleep_fn=lambda _: None,
        on_progress=broken,
    )

    assert results[0].status == "ok"


def test_merge_with_existing_keeps_prior_sources(config, server):
    prior = fetch_source(spec("prior_ok", f"{server}/ok.html"), config, sleep_fn=lambda _: None)
    write_manifest(build_manifest([prior], config), config)
    current = fetch_source(spec("new_ok", f"{server}/ok.html"), config, sleep_fn=lambda _: None)

    merged = merge_with_existing([current], config)
    assert [item.source_id for item in merged] == ["prior_ok", "new_ok"]
    assert all(item.status == "ok" for item in merged)


def test_run_load_dry_run_does_not_download(config):
    result = run_load(config, inventory_path=CONFIG_PATH.parent / "sources.csv", dry_run=True)
    assert result["dry_run"] is True
    assert len(result["sources"]) == 15
    assert read_manifest(config) is None
