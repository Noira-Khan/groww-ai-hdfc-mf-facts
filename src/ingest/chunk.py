"""Chunk normalized records into metadata-rich, embeddable units."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import Config
from src.ingest.loader import read_manifest

TOKEN_CHAR_RATIO = 4
REQUIRED_TOPICS = (
    "expense_ratio",
    "exit_load",
    "min_sip",
    "elss_lockin",
    "riskometer",
    "benchmark",
    "statements",
    "definitions",
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
TABLE_SEPARATOR = re.compile(r"^\|(?:\s*-{3,}\s*\|)+$")


@dataclass
class Unit:
    text: str
    section: str
    page: int | None
    is_table: bool


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    url: str
    publisher: str
    doc_type: str
    scheme: str
    topic: str
    title: str
    section: str
    page: int | None
    as_on_date: str | None
    text: str
    token_count: int
    text_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / TOKEN_CHAR_RATIO))


def _split_sentences(text: str) -> list[str]:
    sentences = [item.strip() for item in SENTENCE_SPLIT.split(text) if item.strip()]
    return sentences or [text.strip()]


def _tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(_split_sentences(text)):
        if tail and total + len(sentence) + 1 > overlap_chars:
            break
        tail.insert(0, sentence)
        total += len(sentence) + 1
        if total >= overlap_chars:
            break
    if not tail:
        tail = [text[-overlap_chars:]]
    result = " ".join(tail).strip()
    if len(result) > overlap_chars:
        result = result[-overlap_chars:]
        boundary = result.find(" ")
        if 0 < boundary < len(result) - 1:
            result = result[boundary + 1 :]
    return result.strip()


def _split_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return pieces


def _split_table(text: str, max_chars: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(text) <= max_chars:
        return [text]

    header_size = 2 if len(lines) > 1 and TABLE_SEPARATOR.match(lines[1]) else 1
    header = lines[:header_size]
    rows = lines[header_size:]
    prefix = "\n".join(header)
    pieces: list[str] = []
    current = prefix

    for row in rows:
        candidate = f"{current}\n{row}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current != prefix:
            pieces.append(current)
        current = f"{prefix}\n{row}"
        if len(current) > max_chars:
            pieces.extend(_split_text(current, max_chars))
            current = prefix
    if current != prefix:
        pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def _table_unit(text: str, max_chars: int, keep_intact: bool) -> list[str]:
    if keep_intact and len(text) <= max_chars:
        return [text]
    if not text.lstrip().startswith("|"):
        return _split_text(text, max_chars)
    return _split_table(text, max_chars)


def detect_topic(section: str, text: str) -> str:
    section_lower = section.lower()
    combined = f"{section} {text}".lower()

    section_rules = (
        ("exit_load", ("exit load",)),
        ("expense_ratio", ("expense", " ter")),
        ("elss_lockin", ("lock-in", "lock in")),
        ("min_sip", ("sip",)),
        ("riskometer", ("riskometer",)),
        ("benchmark", ("benchmark",)),
        ("statements", ("statement",)),
        ("definitions", ("definition", "glossary")),
    )
    for topic, needles in section_rules:
        if any(needle in section_lower for needle in needles):
            return topic

    text_rules = (
        ("expense_ratio", (r"\bexpense ratio\b", r"\bter\b")),
        ("exit_load", (r"\bexit load\b",)),
        ("elss_lockin", (r"\block[\s-]?in\b",)),
        ("min_sip", (r"\bminimum sip\b", r"\bmin sip\b")),
        ("riskometer", (r"\briskometer\b",)),
        ("benchmark", (r"\bbenchmark\b",)),
        ("statements", (r"\baccount statement\b", r"\bcapital[- ]gains? statement\b")),
        (
            "definitions",
            (
                r"\bnet asset value\b",
                r"\bnav\b",
                r"\bsystematic investment plan\b",
                r"\bstands for\b",
                r"\bis defined as\b",
                r"\brefers to\b",
            ),
        ),
        ("scheme_documents", (r"\bkey information memorandum\b", r"\bscheme information document\b")),
    )
    for topic, patterns in text_rules:
        if any(re.search(pattern, combined) for pattern in patterns):
            return topic
    return "general"


def _chunk_id(source_id: str, section: str, text: str) -> str:
    payload = f"{source_id}|{section}|{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _make_chunk(record: dict, section: str, page: int | None, text: str) -> Chunk:
    text = text.strip()
    return Chunk(
        chunk_id=_chunk_id(record["source_id"], section, text),
        source_id=record["source_id"],
        url=record["url"],
        publisher=record["publisher"],
        doc_type=record["doc_type"],
        scheme=record["scheme"],
        topic=detect_topic(section, text),
        title=record.get("title") or "",
        section=section,
        page=page,
        as_on_date=record.get("as_on_date"),
        text=text,
        token_count=estimate_tokens(text),
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _build_units(
    records: list[dict], max_chars: int, unit_max_chars: int, keep_tables_intact: bool
) -> list[Unit]:
    units: list[Unit] = []
    for record in records:
        section = " / ".join(record.get("section_path") or []) or "Document"
        page = record.get("page")
        text = (record.get("text") or "").strip()
        if not text:
            continue
        is_table = text.startswith("|")
        if is_table:
            pieces = _table_unit(text, max_chars, keep_tables_intact)
        else:
            pieces = _split_text(text, unit_max_chars)
        units.extend(Unit(piece, section, page, is_table) for piece in pieces)
    return units


def _section_chunks(
    units: list[Unit], record: dict, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer = ""
    buffer_section = ""
    buffer_page: int | None = None

    def flush() -> str:
        nonlocal buffer, buffer_section, buffer_page
        previous = buffer
        if previous:
            chunks.append(_make_chunk(record, buffer_section, buffer_page, previous))
        buffer = ""
        buffer_section = ""
        buffer_page = None
        return previous

    for unit in units:
        if unit.is_table:
            flush()
            chunks.append(_make_chunk(record, unit.section, unit.page, unit.text))
            continue

        if buffer and (buffer_section != unit.section or buffer_page != unit.page):
            flush()
        candidate = f"{buffer} {unit.text}".strip() if buffer else unit.text
        if buffer and len(candidate) > max_chars:
            previous = flush()
            carry = _tail_overlap(previous, overlap_chars)
            excess = len(carry) + len(unit.text) + 1 - max_chars
            if excess > 0:
                carry = carry[excess:].lstrip()
            buffer = f"{carry} {unit.text}".strip()
            buffer_section = unit.section
            buffer_page = unit.page
        else:
            buffer = candidate
            if not buffer_section:
                buffer_section = unit.section
                buffer_page = unit.page

    flush()
    return chunks


def _fixed_chunks(
    units: list[Unit], record: dict, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    text = " ".join(unit.text for unit in units).strip()
    if not text:
        return []
    step = max(1, max_chars - overlap_chars)
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return [_make_chunk(record, "Document", None, piece) for piece in pieces]


def _choose_strategy(records: list[dict], planner: str) -> tuple[str, str]:
    if planner == "fixed":
        return "fixed_window", "Planner was forced to fixed_window by config."
    if planner in {"section_aware", "recursive"}:
        return (
            "section_aware_recursive",
            "Planner was forced to section_aware_recursive by config.",
        )

    tables = sum(1 for record in records if (record.get("text") or "").startswith("|"))
    sections = {tuple(record.get("section_path") or []) for record in records}
    if tables:
        return (
            "table_preserving",
            f"{tables} table record(s) detected; tables are kept intact with repeated headers.",
        )
    if len(sections) > 1:
        return (
            "section_aware_recursive",
            f"{len(sections)} sections detected across {len(records)} records.",
        )
    return (
        "fixed_window",
        "No tables or usable headings detected; text is split into fixed windows.",
    )


def _corpus_version(config: Config) -> str:
    manifest = read_manifest(config) or {}
    pairs = sorted(
        (row.get("source_id", ""), row.get("sha256", ""))
        for row in manifest.get("sources", [])
        if row.get("status") == "ok"
    )
    if not pairs:
        return "unknown"
    payload = "\n".join(f"{source_id}:{digest}" for source_id, digest in pairs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def _build_stats(chunks: list[Chunk], plan: dict, corpus_version: str) -> dict:
    by_source: dict[str, dict] = defaultdict(lambda: {"chunks": 0, "tokens": 0, "chars": 0})
    by_scheme: Counter[str] = Counter()
    by_topic: Counter[str] = Counter()
    topic_sources: dict[str, set[str]] = defaultdict(set)

    for chunk in chunks:
        entry = by_source[chunk.source_id]
        entry["chunks"] += 1
        entry["tokens"] += chunk.token_count
        entry["chars"] += len(chunk.text)
        by_scheme[chunk.scheme] += 1
        by_topic[chunk.topic] += 1
        topic_sources[chunk.topic].add(chunk.source_id)

    tokens = [chunk.token_count for chunk in chunks]
    buckets = (("<=100", 100), ("101-200", 200), ("201-300", 300), ("301-500", 500), ("500+", None))
    histogram = {label: 0 for label, _ in buckets}
    for count in tokens:
        for label, upper in buckets:
            if upper is None or count <= upper:
                histogram[label] += 1
                break

    coverage = {
        topic: {
            "chunks": by_topic.get(topic, 0),
            "sources": sorted(topic_sources.get(topic, set())),
        }
        for topic in REQUIRED_TOPICS
    }
    coverage["general"] = {
        "chunks": by_topic.get("general", 0),
        "sources": sorted(topic_sources.get("general", set())),
    }
    coverage["scheme_documents"] = {
        "chunks": by_topic.get("scheme_documents", 0),
        "sources": sorted(topic_sources.get("scheme_documents", set())),
    }

    max_tokens = plan["config"]["max_seq_tokens"]
    return {
        "generated_at": _now(),
        "corpus_version": corpus_version,
        "total_chunks": len(chunks),
        "total_chars": sum(len(chunk.text) for chunk in chunks),
        "total_tokens": sum(tokens),
        "token_count": {
            "min": min(tokens) if tokens else 0,
            "max": max(tokens) if tokens else 0,
            "avg": round(sum(tokens) / len(tokens), 2) if tokens else 0,
        },
        "token_histogram": histogram,
        "max_seq_tokens": max_tokens,
        "oversize_chunks": sum(1 for count in tokens if count > max_tokens),
        "by_source": {key: by_source[key] for key in sorted(by_source)},
        "by_scheme": {key: by_scheme[key] for key in sorted(by_scheme)},
        "by_topic": {key: by_topic[key] for key in sorted(by_topic)},
        "topic_coverage": coverage,
        "empty_chunks": sum(1 for chunk in chunks if not chunk.text.strip()),
        "duplicate_chunk_ids": len(chunks) - len({chunk.chunk_id for chunk in chunks}),
        "duplicate_chunks_removed": sum(
            source.get("duplicate_chunks", 0) for source in plan["sources"]
        ),
    }


def run_chunk(
    config: Config,
    normalized_dir: str | Path | None = None,
    only: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    source_dir = (
        Path(normalized_dir)
        if normalized_dir is not None
        else config.app.artifact_dir / "normalized"
    )
    if not source_dir.is_absolute():
        source_dir = config.project_root / source_dir
    if not source_dir.exists():
        raise ValueError("No normalized directory found. Run `--stage normalize` first.")

    files = sorted(source_dir.glob("*.jsonl"))
    if only:
        wanted = {item.strip() for item in only if item.strip()}
        files = [path for path in files if path.stem in wanted]
        if not files:
            raise ValueError(f"No normalized files matched --only: {', '.join(sorted(wanted))}")
    if not files:
        raise ValueError("No normalized jsonl files found. Run `--stage normalize` first.")

    requested_target = config.chunking.target_tokens
    max_seq_tokens = config.embedding.max_seq_tokens
    context_header_reserve = min(32, max(8, max_seq_tokens // 8))
    effective_target = min(requested_target, max(64, max_seq_tokens - context_header_reserve))
    if effective_target < requested_target:
        effective_overlap = min(config.chunking.overlap_tokens, max(8, effective_target // 5))
    else:
        effective_overlap = config.chunking.overlap_tokens
    effective_overlap = min(effective_overlap, max(0, effective_target - 1))
    max_chars = effective_target * TOKEN_CHAR_RATIO
    overlap_chars = effective_overlap * TOKEN_CHAR_RATIO
    unit_max_chars = max(200, max_chars - overlap_chars)

    corpus_version = _corpus_version(config)
    plan_sources: list[dict] = []
    all_chunks: list[Chunk] = []
    strategy_counts: Counter[str] = Counter()

    for path in files:
        records = _load_records(path)
        if not records:
            continue
        strategy, rationale = _choose_strategy(records, config.chunking.planner)
        strategy_counts[strategy] += 1
        units = _build_units(
            records,
            max_chars=max_chars,
            unit_max_chars=unit_max_chars,
            keep_tables_intact=config.chunking.keep_tables_intact,
        )
        if strategy == "fixed_window":
            chunks = _fixed_chunks(units, records[0], max_chars, overlap_chars)
        else:
            chunks = _section_chunks(units, records[0], max_chars, overlap_chars)

        unique: dict[str, Chunk] = {}
        for chunk in chunks:
            unique.setdefault(chunk.chunk_id, chunk)
        duplicates = len(chunks) - len(unique)
        chunks = list(unique.values())

        all_chunks.extend(chunks)
        plan_sources.append(
            {
                "source_id": path.stem,
                "strategy": strategy,
                "rationale": rationale,
                "records": len(records),
                "sections": len({tuple(record.get("section_path") or []) for record in records}),
                "tables": sum(
                    1 for record in records if (record.get("text") or "").startswith("|")
                ),
                "duplicate_chunks": duplicates,
                "chunks": len(chunks),
                "chars": sum(len(chunk.text) for chunk in chunks),
            }
        )

    all_chunks.sort(key=lambda chunk: (chunk.source_id, chunk.section, chunk.chunk_id))
    plan = {
        "generated_at": _now(),
        "corpus_version": corpus_version,
        "planner": config.chunking.planner,
        "config": {
            "requested_target_tokens": requested_target,
            "requested_overlap_tokens": config.chunking.overlap_tokens,
            "effective_target_tokens": effective_target,
            "effective_overlap_tokens": effective_overlap,
            "max_seq_tokens": max_seq_tokens,
            "context_header_reserve_tokens": context_header_reserve,
            "target_chars": max_chars,
            "overlap_chars": overlap_chars,
            "keep_tables_intact": config.chunking.keep_tables_intact,
            "cap_reason": (
                "Target capped by embedding max_seq_tokens, reserving "
                f"{context_header_reserve} tokens for the context header."
                if effective_target < requested_target
                else "Configured target is within embedding limit."
            ),
        },
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "sources": plan_sources,
    }
    stats = _build_stats(all_chunks, plan, corpus_version)

    if dry_run:
        return {
            "dry_run": True,
            "normalized_dir": str(source_dir),
            "files": [path.stem for path in files],
            "plan": plan,
            "stats": stats,
        }

    out_dir = config.app.artifact_dir / "chunks"
    _write_jsonl(all_chunks, out_dir / "chunks.jsonl")
    _write_json(out_dir / "chunking_plan.json", plan)
    _write_json(out_dir / "chunk_stats.json", stats)

    return {
        "chunks_path": (out_dir / "chunks.jsonl").relative_to(config.project_root).as_posix()
        if (out_dir / "chunks.jsonl").is_relative_to(config.project_root)
        else (out_dir / "chunks.jsonl").as_posix(),
        "plan_path": (out_dir / "chunking_plan.json").relative_to(config.project_root).as_posix()
        if (out_dir / "chunking_plan.json").is_relative_to(config.project_root)
        else (out_dir / "chunking_plan.json").as_posix(),
        "stats_path": (out_dir / "chunk_stats.json").relative_to(config.project_root).as_posix()
        if (out_dir / "chunk_stats.json").is_relative_to(config.project_root)
        else (out_dir / "chunk_stats.json").as_posix(),
        "summary": {
            "sources": len(plan_sources),
            "chunks": len(all_chunks),
            "tokens": stats["total_tokens"],
            "oversize": stats["oversize_chunks"],
            "topics": len([item for item in stats["topic_coverage"].values() if item["chunks"]]),
        },
        "config": plan["config"],
        "strategy_counts": plan["strategy_counts"],
        "topic_coverage": stats["topic_coverage"],
    }
