"""Download allowlisted source documents and record a manifest."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from src.config import Config
from src.ingest.inventory import SourceSpec, load_inventory, validate_inventory
from src.logging import get_logger, safe_log

MAX_BYTES = 40 * 1024 * 1024
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_BACKOFF_S = 30.0
FetchProgress = Callable[[int, int, "FetchResult"], None]


class RetryableStatus(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"retryable status {status_code}")
        self.status_code = status_code


@dataclass
class FetchResult:
    source_id: str
    url: str
    status: str
    http_status: int | None = None
    content_type: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    fetched_at: str | None = None
    local_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _timeout(config: Config) -> httpx.Timeout:
    """Fail fast on unreachable hosts while keeping the configured read timeout."""
    read_timeout = float(config.ingestion.request_timeout_s)
    return httpx.Timeout(read_timeout, connect=min(10.0, read_timeout))


def _safe_name(source_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in source_id)
    return cleaned.strip("_") or "source"


def _detect_extension(data: bytes, content_type: str) -> str:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if "pdf" in content_type.lower():
        return "pdf"
    return "html"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _failed(spec: SourceSpec, error: str, http_status: int | None = None) -> FetchResult:
    return FetchResult(
        source_id=spec.source_id,
        url=spec.url,
        status="failed",
        http_status=http_status,
        fetched_at=_now(),
        error=error,
    )


def fetch_source(
    spec: SourceSpec,
    config: Config,
    client: httpx.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> FetchResult:
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=_timeout(config),
            follow_redirects=True,
            headers={"User-Agent": config.ingestion.user_agent},
        )

    attempts = max(1, config.ingestion.max_retries)
    response: httpx.Response | None = None
    data = b""
    last_error = "unknown error"

    try:
        for attempt in range(1, attempts + 1):
            try:
                response = client.get(spec.url)
                if response.status_code in RETRYABLE_STATUS:
                    raise RetryableStatus(response.status_code)
                if response.status_code >= 400:
                    return _failed(spec, f"HTTP {response.status_code}", response.status_code)
                data = response.content
                if len(data) > MAX_BYTES:
                    return _failed(spec, f"response exceeds {MAX_BYTES} bytes", response.status_code)
                break
            except (httpx.TransportError, RetryableStatus) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= attempts:
                    return _failed(spec, last_error)
                backoff = (config.ingestion.request_delay_ms / 1000) * (
                    2 ** (attempt - 1)
                )
                sleep_fn(min(MAX_BACKOFF_S, backoff))
            except httpx.HTTPError as exc:
                return _failed(spec, f"{type(exc).__name__}: {exc}")

        if response is None:
            return _failed(spec, last_error)

        content_type = response.headers.get("content-type", "application/octet-stream")
        extension = _detect_extension(data, content_type)
        raw_dir = config.app.artifact_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        local_path = raw_dir / f"{_safe_name(spec.source_id)}.{extension}"
        local_path.write_bytes(data)

        return FetchResult(
            source_id=spec.source_id,
            url=spec.url,
            status="ok",
            http_status=response.status_code,
            content_type=content_type,
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            fetched_at=_now(),
            local_path=_relative(local_path, config.project_root),
        )
    finally:
        if owns_client:
            client.close()


def fetch_all(
    specs: list[SourceSpec],
    config: Config,
    fail_fast: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_progress: FetchProgress | None = None,
) -> list[FetchResult]:
    results: list[FetchResult] = []
    logger = get_logger("ingest.loader")

    with httpx.Client(
        timeout=_timeout(config),
        follow_redirects=True,
        headers={"User-Agent": config.ingestion.user_agent},
    ) as client:
        for index, spec in enumerate(specs):
            try:
                result = fetch_source(spec, config, client=client, sleep_fn=sleep_fn)
            except Exception as exc:
                result = _failed(spec, f"{type(exc).__name__}: {exc}")

            results.append(result)
            safe_log(
                logger,
                "source_fetched",
                source_id=result.source_id,
                status=result.status,
                http_status=result.http_status,
                bytes=result.bytes,
            )
            if on_progress is not None:
                try:
                    on_progress(len(results), len(specs), result)
                except Exception:
                    # Progress reporting must never break ingestion.
                    pass

            if result.status != "ok" and fail_fast:
                break
            if index < len(specs) - 1:
                sleep_fn(config.ingestion.request_delay_ms / 1000)

    return results


def build_manifest(results: list[FetchResult], config: Config) -> dict:
    summary = {
        "total": len(results),
        "ok": sum(1 for item in results if item.status == "ok"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "bytes": sum(item.bytes or 0 for item in results if item.status == "ok"),
    }
    return {
        "generated_at": _now(),
        "publishers": config.ingestion.allowlist_publishers,
        "summary": summary,
        "sources": [item.to_dict() for item in results],
    }


def write_manifest(manifest: dict, config: Config) -> Path:
    raw_dir = config.app.artifact_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_manifest(config: Config) -> dict | None:
    path = config.app.artifact_dir / "raw" / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def merge_with_existing(results: list[FetchResult], config: Config) -> list[FetchResult]:
    """Keep prior results when re-running a subset with --only."""
    existing = read_manifest(config)
    if not existing:
        return results

    fields = FetchResult.__dataclass_fields__
    updated = {item.source_id: item for item in results}
    merged: list[FetchResult] = []
    seen: set[str] = set()

    for row in existing.get("sources", []):
        source_id = row.get("source_id")
        if not source_id:
            continue
        if source_id in updated:
            merged.append(updated[source_id])
            seen.add(source_id)
        else:
            merged.append(FetchResult(**{key: row.get(key) for key in fields}))

    for item in results:
        if item.source_id not in seen:
            merged.append(item)

    return merged


def run_load(
    config: Config,
    inventory_path: str | Path = "sources.csv",
    only: list[str] | None = None,
    dry_run: bool = False,
    fail_fast: bool = False,
    on_progress: FetchProgress | None = None,
) -> dict:
    inventory_file = Path(inventory_path)
    if not inventory_file.is_absolute():
        inventory_file = config.project_root / inventory_file

    specs = load_inventory(inventory_file)
    validate_inventory(specs, config.ingestion.allowlist_publishers)

    if only:
        wanted = {item.strip() for item in only if item.strip()}
        specs = [spec for spec in specs if spec.source_id in wanted]
        if not specs:
            raise ValueError(f"No inventory rows matched --only: {', '.join(sorted(wanted))}")

    if dry_run:
        return {
            "dry_run": True,
            "inventory": str(inventory_file),
            "sources": [asdict(spec) for spec in specs],
        }

    results = fetch_all(
        specs, config, fail_fast=fail_fast, on_progress=on_progress
    )
    if only:
        results = merge_with_existing(results, config)
    manifest = build_manifest(results, config)
    manifest["inventory"] = _relative(inventory_file, config.project_root)
    path = write_manifest(manifest, config)
    manifest["manifest_path"] = _relative(path, config.project_root)
    return manifest
