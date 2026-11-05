"""Source inventory parsing and validation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_DOC_TYPES = {
    "scheme_page",
    "kim",
    "sid",
    "factsheet",
    "ter",
    "statement_guide",
    "faq",
    "catalogue",
    "education",
    "groww_page",
}

ALLOWED_SCHEMES = {"large_cap", "flexi_cap", "elss", "mid_cap", "small_cap", "amc_wide"}

ALLOWED_TOPICS = {
    "expense_ratio",
    "exit_load",
    "min_sip",
    "elss_lockin",
    "riskometer",
    "benchmark",
    "statements",
    "definitions",
    "scheme_documents",
    "general",
}

PUBLISHER_DOMAINS = {
    "HDFC AMC": ("hdfcfund.com",),
    "SEBI": ("sebi.gov.in",),
    "AMFI": ("amfiindia.com",),
    "Groww": ("groww.in", "groww.com"),
}

SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9_]{3,64}$")

REQUIRED_COLUMNS = (
    "source_id",
    "url",
    "publisher",
    "doc_type",
    "scheme",
    "title",
    "expected_topics",
)


@dataclass
class SourceSpec:
    source_id: str
    url: str
    publisher: str
    doc_type: str
    scheme: str
    title: str
    expected_topics: list[str] = field(default_factory=list)


def _split_topics(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[|;]", value or "") if item.strip()]


def load_inventory(path: str | Path) -> list[SourceSpec]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source inventory not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

        specs: list[SourceSpec] = []
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                specs.append(
                    SourceSpec(
                        source_id=(row["source_id"] or "").strip(),
                        url=(row["url"] or "").strip(),
                        publisher=(row["publisher"] or "").strip(),
                        doc_type=(row["doc_type"] or "").strip(),
                        scheme=(row["scheme"] or "").strip(),
                        title=(row["title"] or "").strip(),
                        expected_topics=_split_topics(row.get("expected_topics", "")),
                    )
                )
            except Exception as exc:
                raise ValueError(f"{path} row {row_number}: {exc}") from exc

    return specs


def _host_allowed(host: str, domains: tuple[str, ...]) -> bool:
    host = host.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def validate_inventory(specs: list[SourceSpec], allowlist_publishers: list[str]) -> None:
    if not specs:
        raise ValueError("Source inventory is empty")
    if not allowlist_publishers:
        raise ValueError("No publishers are allowlisted")

    allowlist = {publisher.strip() for publisher in allowlist_publishers}
    seen: set[str] = set()

    for spec in specs:
        location = f"source '{spec.source_id}'"

        if not SOURCE_ID_PATTERN.match(spec.source_id):
            raise ValueError(f"{location}: source_id must match {SOURCE_ID_PATTERN.pattern}")

        if spec.source_id in seen:
            raise ValueError(f"{location}: duplicate source_id")
        seen.add(spec.source_id)

        if spec.publisher not in allowlist:
            raise ValueError(f"{location}: publisher '{spec.publisher}' is not allowlisted")

        domains = PUBLISHER_DOMAINS.get(spec.publisher)
        if not domains:
            raise ValueError(f"{location}: no trusted domains configured for '{spec.publisher}'")

        parsed = urlparse(spec.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{location}: url must be an absolute http(s) URL")

        if not _host_allowed(parsed.hostname or "", domains):
            raise ValueError(
                f"{location}: host '{parsed.hostname}' does not belong to {spec.publisher}"
            )

        if spec.doc_type not in ALLOWED_DOC_TYPES:
            raise ValueError(
                f"{location}: doc_type '{spec.doc_type}' must be one of "
                f"{', '.join(sorted(ALLOWED_DOC_TYPES))}"
            )

        if spec.scheme not in ALLOWED_SCHEMES:
            raise ValueError(
                f"{location}: scheme '{spec.scheme}' must be one of "
                f"{', '.join(sorted(ALLOWED_SCHEMES))}"
            )

        unknown_topics = sorted(set(spec.expected_topics) - ALLOWED_TOPICS)
        if unknown_topics:
            raise ValueError(f"{location}: unknown topics: {', '.join(unknown_topics)}")

        if not spec.title:
            raise ValueError(f"{location}: title must not be empty")


def summarise_inventory(specs: list[SourceSpec]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for spec in specs:
        summary[spec.publisher] = summary.get(spec.publisher, 0) + 1
        summary[f"doc_type:{spec.doc_type}"] = summary.get(f"doc_type:{spec.doc_type}", 0) + 1
    return summary
