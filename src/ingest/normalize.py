"""Normalize raw HTML and PDF sources into structured text records."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.config import Config
from src.ingest.inventory import SourceSpec, load_inventory, validate_inventory
from src.ingest.loader import FetchResult, read_manifest
from src.logging import get_logger, safe_log

MIN_TEXT_LEN = 20
MAX_TEXT_LEN = 8000
HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "table"]
REMOVE_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "button", "input", "select", "textarea"]

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

MONTH_WORD = (
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
DATE_MDY = re.compile(rf"\b{MONTH_WORD}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
DATE_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{MONTH_WORD},?\s+(\d{{4}})\b", re.I)
DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_SLASH = re.compile(r"\b(\d{1,2})[/](\d{1,2})[/](\d{4})\b")

BOILERPLATE_PREFIXES = (
    "skip to content",
    "accept all cookies",
    "this website uses cookies",
    "download the app",
    "follow us on",
)


@dataclass
class TextRecord:
    source_id: str
    url: str
    publisher: str
    doc_type: str
    scheme: str
    title: str
    as_on_date: str | None
    section_path: list[str]
    page: int | None
    text: str
    fetched_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _clean_block(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def extract_date(text: str) -> str | None:
    sample = text or ""

    for match in DATE_ISO.finditer(sample):
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).date().isoformat()
        except ValueError:
            continue

    for match in DATE_MDY.finditer(sample):
        month = MONTHS.get(match.group(1)[:3].lower())
        if month:
            try:
                return datetime(int(match.group(3)), month, int(match.group(2))).date().isoformat()
            except ValueError:
                continue

    for match in DATE_DMY.finditer(sample):
        month = MONTHS.get(match.group(2)[:3].lower())
        if month:
            try:
                return datetime(int(match.group(3)), month, int(match.group(1))).date().isoformat()
            except ValueError:
                continue

    for match in DATE_SLASH.finditer(sample):
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return datetime(year, month, day).date().isoformat()
            except ValueError:
                continue

    return None


def _decode_html(data: bytes, content_type: str | None) -> str:
    match = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    encodings = [match.group(1)] if match else []
    encodings.extend(["utf-8", "cp1252"])
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _is_boilerplate(text: str) -> bool:
    lowered = text.lower()
    if len(text) > MAX_TEXT_LEN:
        return True
    if lowered in BOILERPLATE_PREFIXES:
        return True
    return any(lowered.startswith(prefix) for prefix in BOILERPLATE_PREFIXES)


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for tag in soup(REMOVE_TAGS):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "header"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if not tag.attrs:
            continue
        identifier = " ".join(
            [
                " ".join(tag.get("class", []) or []),
                str(tag.get("id", "") or ""),
                str(tag.get("role", "") or ""),
            ]
        ).lower()
        if any(
            hint in identifier
            for hint in ("cookie", "breadcrumb", "popup", "modal", "advert", "newsletter")
        ):
            tag.decompose()


def _table_to_markdown(table) -> str:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        values = [_clean_block(cell.get_text(" ", strip=True))[:160] for cell in cells]
        if any(values):
            rows.append(values)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    if len(lines) >= 1:
        lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    text = "\n".join(lines)
    return text if len(text) >= MIN_TEXT_LEN else ""


def _main_soup(html: str) -> BeautifulSoup:
    try:
        from readability import Document

        summary = Document(html).summary()
        candidate = BeautifulSoup(summary, "lxml")
        if len(_clean_block(candidate.get_text(" ", strip=True))) >= 400:
            return candidate
    except Exception:
        pass
    return BeautifulSoup(html, "lxml")


FACT_LABELS = {
    "inception date": "Inception Date",
    "riskometer": "Riskometer",
    "min sip": "Minimum SIP",
    "ideal for": "Ideal For",
    "entry load": "Entry Load",
    "ter": "Total Expense Ratio (TER)",
    "lock in": "Lock-in",
    "lock-in": "Lock-in",
    "nav": "NAV",
    "aum": "Assets Under Management (AUM)",
}


def _records_from_fact_cards(
    soup: BeautifulSoup,
    spec: SourceSpec,
    as_on_date: str | None,
    fetched_at: str | None,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    for value_node in soup.find_all("p"):
        classes = " ".join(value_node.get("class", []) or []).lower()
        if "description" not in classes:
            continue
        parent = value_node.parent
        if parent is None:
            continue
        title_node = parent.find(
            lambda tag: tag.name in {"p", "div", "span"}
            and "title" in " ".join(tag.get("class", []) or []).lower()
        )
        if title_node is None:
            continue
        raw_label = next(title_node.stripped_strings, "")
        label = FACT_LABELS.get(raw_label.strip().lower())
        value = _clean_block(value_node.get_text(" ", strip=True))
        if not label or not value or value.upper() == "NA":
            continue
        if label == "Total Expense Ratio (TER)" and re.fullmatch(r"\d+(?:\.\d+)?", value):
            value = f"{value}%"
        text = f"{label}: {value}"
        records.append(
            _make_record(spec, as_on_date, fetched_at, ["Scheme Facts", label], None, text)
        )
    return records


def _make_record(
    spec: SourceSpec,
    as_on_date: str | None,
    fetched_at: str | None,
    section: list[str],
    page: int | None,
    text: str,
) -> TextRecord:
    return TextRecord(
        source_id=spec.source_id,
        url=spec.url,
        publisher=spec.publisher,
        doc_type=spec.doc_type,
        scheme=spec.scheme,
        title=spec.title,
        as_on_date=as_on_date,
        section_path=list(section),
        page=page,
        text=text,
        fetched_at=fetched_at,
    )


def _records_from_soup(
    soup: BeautifulSoup,
    spec: SourceSpec,
    as_on_date: str | None,
    fetched_at: str | None,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    section: list[str] = []

    for element in soup.find_all(BLOCK_TAGS):
        if element.name != "table" and element.find_parent("table"):
            continue

        if element.name in HEADING_TAGS:
            text = _clean_block(element.get_text(" ", strip=True))
            if not text:
                continue
            level = HEADING_TAGS[element.name]
            section = section[: level - 1] + [text]
            continue

        if element.name == "table":
            text = _table_to_markdown(element)
        else:
            text = _clean_block(element.get_text(" ", strip=True))

        if len(text) < MIN_TEXT_LEN or _is_boilerplate(text):
            continue
        records.append(_make_record(spec, as_on_date, fetched_at, section, None, text))

    return records


def _groww_risk_label(mf: dict) -> str | None:
    """Prefer live risk from return stats over stale NFO risk labels."""
    for item in mf.get("return_stats") or []:
        if isinstance(item, dict):
            risk = item.get("risk")
            if isinstance(risk, str) and risk.strip():
                return risk.strip()
    for key in ("risk", "riskometer"):
        value = mf.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nfo = mf.get("nfo_risk")
    if isinstance(nfo, str) and nfo.strip():
        return re.sub(r"\s*riskometer\s*$", "", nfo.strip(), flags=re.I).strip()
    return None


def _groww_lock_in_years(mf: dict) -> int | None:
    lock = mf.get("lock_in")
    if isinstance(lock, dict):
        years = lock.get("years")
        if isinstance(years, (int, float)) and years > 0:
            return int(years)
    for key in ("lock_in_yrs", "lockin_period", "lock_in_period"):
        value = mf.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def _records_from_groww_next_data(
    html: str,
    spec: SourceSpec,
    as_on_date: str | None,
    fetched_at: str | None,
) -> list[TextRecord]:
    """Extract scheme facts from Groww's embedded __NEXT_DATA__ payload."""
    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html,
        re.S,
    )
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    mf = (
        payload.get("props", {})
        .get("pageProps", {})
        .get("mfServerSideData")
    )
    if not isinstance(mf, dict) or not mf:
        return []

    lines: list[str] = []
    scheme_name = mf.get("scheme_name")
    if isinstance(scheme_name, str) and scheme_name.strip():
        lines.append(f"Scheme: {scheme_name.strip()}")

    risk = _groww_risk_label(mf)
    if risk:
        lines.append(f"Riskometer: {risk}")

    benchmark = mf.get("benchmark") or mf.get("benchmark_name")
    if isinstance(benchmark, str) and benchmark.strip():
        lines.append(f"Benchmark: {benchmark.strip()}")

    expense = mf.get("expense_ratio")
    if expense is not None and str(expense).strip():
        value = str(expense).strip()
        if not value.endswith("%"):
            value = f"{value}%"
        lines.append(f"Total Expense Ratio (TER): {value}")

    exit_load = mf.get("exit_load")
    if isinstance(exit_load, str) and exit_load.strip():
        text = exit_load.strip()
        if not text.lower().startswith("exit load"):
            text = f"Exit load: {text}"
        elif not text.lower().startswith("exit load:"):
            text = re.sub(r"^exit load\b", "Exit load:", text, count=1, flags=re.I)
        lines.append(text)

    min_sip = mf.get("min_sip_investment")
    if isinstance(min_sip, (int, float)) and min_sip > 0:
        amount = int(min_sip) if float(min_sip).is_integer() else min_sip
        lines.append(f"Minimum SIP: ₹ {amount}")

    lock_years = _groww_lock_in_years(mf)
    if lock_years:
        lines.append(f"Lock-in: {lock_years} years")

    if not lines:
        return []

    # One combined card plus individual lines so topic retrieval can match cleanly.
    records = [
        _make_record(
            spec,
            as_on_date,
            fetched_at,
            ["Scheme Facts"],
            None,
            "\n".join(lines),
        )
    ]
    for line in lines:
        label = line.split(":", 1)[0].strip()
        records.append(
            _make_record(
                spec,
                as_on_date,
                fetched_at,
                ["Scheme Facts", label],
                None,
                line,
            )
        )
    return records


def normalize_html(
    data: bytes,
    spec: SourceSpec,
    content_type: str | None = None,
    fetched_at: str | None = None,
) -> list[TextRecord]:
    html = _decode_html(data, content_type)
    full_soup = BeautifulSoup(html, "lxml")
    _strip_boilerplate(full_soup)
    full_text = _clean_block(full_soup.get_text(" ", strip=True))
    as_on_date = extract_date(full_text[:20000])
    # Groww marketing copy often embeds unrelated calendar dates; prefer fetch day.
    if spec.doc_type == "groww_page" and fetched_at:
        try:
            as_on_date = datetime.fromisoformat(
                fetched_at.replace("Z", "+00:00")
            ).date().isoformat()
        except ValueError:
            pass

    records: list[TextRecord] = []
    if spec.doc_type == "groww_page":
        records.extend(
            _records_from_groww_next_data(html, spec, as_on_date, fetched_at)
        )

    soup = _main_soup(html)
    _strip_boilerplate(soup)
    records.extend(_records_from_soup(soup, spec, as_on_date, fetched_at))
    records.extend(_records_from_fact_cards(full_soup, spec, as_on_date, fetched_at))

    existing = {record.text for record in records}
    # Skip return/performance tables on Groww pages; facts come from NEXT_DATA.
    if spec.doc_type != "groww_page":
        for table in full_soup.find_all("table"):
            markdown = _table_to_markdown(table)
            if markdown and markdown not in existing:
                records.append(
                    _make_record(spec, as_on_date, fetched_at, ["Tables"], None, markdown)
                )
                existing.add(markdown)

    return dedupe_records(records)


def _is_pdf_heading(text: str) -> bool:
    if len(text) > 80:
        return False
    words = text.split()
    if not words or len(words) > 12:
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    if not any(char.isalpha() for char in text):
        return False
    return text.isupper() or text[0].isupper()


def records_from_pdf_pages(
    pages: list[str],
    spec: SourceSpec,
    as_on_date: str | None,
    fetched_at: str | None,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    section: list[str] = []

    for page_number, page_text in enumerate(pages, start=1):
        lines = [clean_text(line) for line in (page_text or "").splitlines()]
        lines = [line for line in lines if line]
        paragraph: list[str] = []

        def flush() -> None:
            nonlocal paragraph
            if not paragraph:
                return
            text = _clean_block(" ".join(paragraph))
            paragraph = []
            if len(text) < MIN_TEXT_LEN or _is_boilerplate(text):
                return
            records.append(
                _make_record(spec, as_on_date, fetched_at, section, page_number, text)
            )

        for line in lines:
            if _is_pdf_heading(line):
                flush()
                section = [line]
                continue
            paragraph.append(line)
            if len(" ".join(paragraph)) >= 900:
                flush()

        flush()

    return records


def pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def dedupe_records(records: list[TextRecord]) -> list[TextRecord]:
    seen: set[tuple[str, str, int | None]] = set()
    output: list[TextRecord] = []

    for record in records:
        digest = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        key = (" / ".join(record.section_path), digest, record.page)
        if key in seen:
            continue
        seen.add(key)
        output.append(record)

    return output


def normalize_source(
    spec: SourceSpec,
    result: FetchResult,
    project_root: Path,
) -> tuple[list[TextRecord], int | None, str | None]:
    if not result.local_path:
        return [], None, "manifest row has no local_path"

    path = Path(result.local_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return [], None, f"raw file missing: {path.name}"

    data = path.read_bytes()
    fetched_at = result.fetched_at

    if data.startswith(b"%PDF-") or path.suffix.lower() == ".pdf":
        pages = pdf_pages(path)
        text = _clean_block(" ".join(pages))
        as_on_date = extract_date(text[:20000])
        records = records_from_pdf_pages(pages, spec, as_on_date, fetched_at)
        return dedupe_records(records), len(pages), None

    html = _decode_html(data, result.content_type)
    text = _clean_block(BeautifulSoup(html, "lxml").get_text(" ", strip=True))
    as_on_date = extract_date(text[:20000])
    records = normalize_html(data, spec, result.content_type, fetched_at)
    return dedupe_records(records), None, None


def _write_jsonl(records: list[TextRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_coverage(config: Config) -> dict | None:
    path = config.app.artifact_dir / "normalized" / "_coverage.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_coverage(entries: list[dict], config: Config) -> list[dict]:
    existing = _read_coverage(config)
    if not existing:
        return entries
    updated = {entry["source_id"]: entry for entry in entries}
    merged: list[dict] = []
    seen: set[str] = set()
    for entry in existing.get("sources", []):
        source_id = entry.get("source_id")
        if source_id in updated:
            merged.append(updated[source_id])
            seen.add(source_id)
        else:
            merged.append(entry)
    for entry in entries:
        if entry["source_id"] not in seen:
            merged.append(entry)
    return merged


def run_normalize(
    config: Config,
    inventory_path: str | Path = "sources.csv",
    only: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    manifest = read_manifest(config)
    if not manifest:
        raise ValueError("No raw manifest found. Run `--stage load` first.")

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

    results = {row["source_id"]: FetchResult(**row) for row in manifest.get("sources", [])}

    if dry_run:
        return {
            "dry_run": True,
            "inventory": str(inventory_file),
            "sources": [
                {"source_id": spec.source_id, "raw_available": results.get(spec.source_id, FetchResult("", "", "missing")).status == "ok"}
                for spec in specs
            ],
        }

    logger = get_logger("ingest.normalize")
    out_dir = config.app.artifact_dir / "normalized"
    coverage: list[dict] = []

    for spec in specs:
        result = results.get(spec.source_id)
        entry = {
            "source_id": spec.source_id,
            "publisher": spec.publisher,
            "doc_type": spec.doc_type,
            "scheme": spec.scheme,
            "status": "missing",
            "records": 0,
            "chars": 0,
            "pages": None,
            "as_on_date": None,
            "error": "no successful fetch in manifest",
        }

        if result and result.status == "ok":
            try:
                records, pages, error = normalize_source(spec, result, config.project_root)
                if error:
                    entry.update(status="unusable", error=error)
                elif not records:
                    entry.update(status="unusable", error="no extractable text")
                else:
                    _write_jsonl(records, out_dir / f"{spec.source_id}.jsonl")
                    entry.update(
                        status="ok",
                        records=len(records),
                        chars=sum(len(record.text) for record in records),
                        pages=pages,
                        as_on_date=records[0].as_on_date,
                        error=None,
                    )
            except Exception as exc:
                entry.update(status="unusable", error=f"{type(exc).__name__}: {exc}")

        coverage.append(entry)
        safe_log(
            logger,
            "source_normalized",
            source_id=entry["source_id"],
            status=entry["status"],
            stage="normalize",
        )

    if only:
        coverage = _merge_coverage(coverage, config)

    summary = {
        "sources": len(coverage),
        "ok": sum(1 for entry in coverage if entry["status"] == "ok"),
        "unusable": sum(1 for entry in coverage if entry["status"] == "unusable"),
        "missing": sum(1 for entry in coverage if entry["status"] == "missing"),
        "records": sum(entry["records"] for entry in coverage),
        "chars": sum(entry["chars"] for entry in coverage),
    }

    payload = {
        "generated_at": _now(),
        "summary": summary,
        "sources": coverage,
    }

    path = out_dir / "_coverage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

    return {
        "coverage_path": path.relative_to(config.project_root).as_posix()
        if path.is_relative_to(config.project_root)
        else path.as_posix(),
        "summary": summary,
        "sources": coverage,
    }
