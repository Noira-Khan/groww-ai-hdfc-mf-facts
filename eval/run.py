"""Run retrieval, QA, and safety evaluations against the gold set.

Examples:
    python -m eval.run --suite all
    python -m eval.run --suite retrieval
    python -m eval.run --suite safety

Writes JSON artifacts under data/eval/. Raw questions, PII values, and secrets
are never written to evaluation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Config, load_config
from src.rag.pipeline import answer_question
from src.rag.retrieve import retrieve
from src.rag.validate import (
    count_sentences,
    extract_urls,
    is_allowlisted_url,
    validate_candidate,
)
from src.safety import responses
from src.safety.intent import classify_question
from src.safety.pii import detect_pii

DEFAULT_GOLD_PATH = Path(__file__).resolve().parent / "gold_questions.csv"
GUARDRAIL_TYPES = {
    "refusal_advice",
    "refusal_performance",
    "pii",
    "not_in_sources",
    "clarify",
}


@dataclass(frozen=True)
class GoldRow:
    id: str
    question: str
    expected_type: str
    expected_scheme: str
    expected_source_id: str
    expected_fact: str
    notes: str

    @property
    def schemes(self) -> list[str]:
        return _split(self.expected_scheme)

    @property
    def source_ids(self) -> list[str]:
        return _split(self.expected_source_id)

    @property
    def facts(self) -> list[str]:
        return _split(self.expected_fact)

    @property
    def primary_scheme(self) -> str | None:
        return self.schemes[0] if self.schemes else None


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_gold(path: Path = DEFAULT_GOLD_PATH) -> list[GoldRow]:
    if not path.exists():
        raise FileNotFoundError(f"Gold set not found: {path}")
    rows: list[GoldRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id",
            "question",
            "expected_type",
            "expected_scheme",
            "expected_source_id",
            "expected_fact",
            "notes",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Gold set is missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            if not (raw.get("id") or "").strip():
                continue
            rows.append(
                GoldRow(
                    id=(raw.get("id") or "").strip(),
                    question=(raw.get("question") or "").strip(),
                    expected_type=(raw.get("expected_type") or "").strip(),
                    expected_scheme=(raw.get("expected_scheme") or "").strip(),
                    expected_source_id=(raw.get("expected_source_id") or "").strip(),
                    expected_fact=(raw.get("expected_fact") or "").strip(),
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    if len(rows) < 25:
        raise ValueError(f"Gold set must contain at least 25 rows; found {len(rows)}")
    return rows


def _source_url_map(config: Config) -> dict[str, set[str]]:
    path = config.project_root / "sources.csv"
    mapping: dict[str, set[str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_id = (row.get("source_id") or "").strip()
            url = (row.get("url") or "").strip()
            if source_id and url:
                mapping.setdefault(source_id, set()).add(url)
    return mapping


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _envelope_urls(envelope: dict[str, Any] | None) -> list[str]:
    if not envelope:
        return []
    urls = extract_urls(str(envelope.get("answer") or ""))
    source_url = envelope.get("source_url")
    if source_url:
        urls.append(str(source_url))
    return list(dict.fromkeys(urls))


def _has_pii(envelope: dict[str, Any] | None) -> bool:
    if not envelope:
        return False
    texts = [envelope.get("answer"), envelope.get("proof")]
    texts.extend(envelope.get("can_do") or [])
    return any(detect_pii(str(text)) for text in texts if text)


def _guardrail_envelope(
    config: Config, question: str
) -> tuple[dict[str, Any] | None, str]:
    if not question:
        return responses.error_response("eval-empty"), "error"

    if config.safety.detect_pii and detect_pii(question):
        return responses.pii_response("eval"), "pii"

    intent = classify_question(question)
    if intent.route == "refusal_advice":
        return responses.advice_response("eval"), intent.route
    if intent.route == "refusal_performance":
        return responses.performance_response("eval"), intent.route
    if intent.route == "not_in_sources":
        return responses.not_in_sources_response("eval"), intent.route
    if intent.route == "clarify":
        return responses.clarify_response(question, "eval"), intent.route
    return None, intent.route


def run_safety(config: Config, rows: list[GoldRow]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    routing_correct = 0
    citation_total = 0
    citation_correct = 0
    pii_leaks = 0

    for row in rows:
        if row.expected_type not in GUARDRAIL_TYPES:
            continue
        envelope, route = _guardrail_envelope(config, row.question)
        actual_type = str((envelope or {}).get("type") or route)
        route_ok = actual_type == row.expected_type
        routing_correct += int(route_ok)

        urls = _envelope_urls(envelope)
        if envelope is not None and actual_type != "error":
            citation_total += 1
            citation_correct += int(
                len(urls) == 1
                and is_allowlisted_url(str(envelope.get("source_url") or ""))
            )

        leak = _has_pii(envelope)
        pii_leaks += int(leak)
        cases.append(
            {
                "id": row.id,
                "expected_type": row.expected_type,
                "actual_type": actual_type,
                "route_ok": route_ok,
                "source_url": (envelope or {}).get("source_url"),
                "url_count": len(urls),
                "suggestion_count": len((envelope or {}).get("can_do") or []),
                "pii_leak": leak,
            }
        )

    total = len(cases)
    routing_accuracy = routing_correct / total if total else 1.0
    citation_precision = citation_correct / citation_total if citation_total else 1.0
    passed = (
        total > 0
        and routing_accuracy == 1.0
        and citation_precision == 1.0
        and pii_leaks == 0
    )
    return {
        "suite": "safety",
        "generated_at": _now(),
        "summary": {
            "total": total,
            "routing_correct": routing_correct,
            "routing_accuracy": round(routing_accuracy, 4),
            "citation_total": citation_total,
            "citation_correct": citation_correct,
            "citation_precision": round(citation_precision, 4),
            "pii_leaks": pii_leaks,
            "pass": passed,
        },
        "cases": cases,
    }


def run_retrieval(config: Config, rows: list[GoldRow], model) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    hits = 0

    for row in rows:
        if row.expected_type != "answer" or not row.source_ids:
            continue
        result = retrieve(
            config,
            row.question,
            scheme=row.primary_scheme,
            model=model,
            persist_debug=False,
        )
        candidates = result.debug.get("candidates", [])
        candidate_ids = [str(item.get("source_id") or "") for item in candidates]
        selected_ids = [
            str(item.get("source_id") or "") for item in result.debug.get("selected", [])
        ]
        hit = any(source_id in candidate_ids for source_id in row.source_ids)
        hits += int(hit)
        cases.append(
            {
                "id": row.id,
                "expected_source_ids": row.source_ids,
                "hit": hit,
                "candidate_source_ids": candidate_ids,
                "selected_source_ids": selected_ids,
                "candidate_scores": [
                    round(float(item.get("score", 0.0)), 4) for item in candidates
                ],
            }
        )

    total = len(cases)
    recall = hits / total if total else 0.0
    return {
        "suite": "retrieval",
        "generated_at": _now(),
        "summary": {
            "total": total,
            "hits": hits,
            "recall_at_5": round(recall, 4),
            "threshold": 0.85,
            "pass": total > 0 and recall >= 0.85,
        },
        "cases": cases,
    }


def run_qa(
    config: Config, rows: list[GoldRow], model, source_urls: dict[str, set[str]]
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    factual_total = 0
    factual_correct = 0
    citation_total = 0
    citation_correct = 0
    sentence_violations = 0
    ungrounded_numbers = 0
    pii_leaks = 0

    for row in rows:
        if row.expected_type != "answer":
            continue

        retrieval = retrieve(
            config,
            row.question,
            scheme=row.primary_scheme,
            model=model,
            persist_debug=False,
        )
        envelope = answer_question(
            config,
            row.question,
            scheme=row.primary_scheme,
            model=model,
            persist_debug=False,
        )

        actual_type = str(envelope.get("type") or "error")
        answer = str(envelope.get("answer") or "")
        proof = str(envelope.get("proof") or "")
        searchable = f"{answer} {proof}".lower()
        fact_hit = any(fact.lower() in searchable for fact in row.facts)
        factual_total += 1
        factual_correct += int(fact_hit)

        expected_urls: set[str] = set()
        for source_id in row.source_ids:
            expected_urls.update(source_urls.get(source_id, set()))
        source_hit = (
            str(envelope.get("source_url") or "") in expected_urls
            if expected_urls
            else None
        )

        urls = _envelope_urls(envelope)
        if actual_type != "error":
            citation_total += 1
            citation_correct += int(
                len(urls) == 1
                and is_allowlisted_url(str(envelope.get("source_url") or ""))
            )

        if actual_type == "answer":
            if count_sentences(answer) > config.app.max_answer_sentences:
                sentence_violations += 1
            if retrieval.chunks:
                validation = validate_candidate(
                    config,
                    {
                        "answer": answer,
                        "source_url": envelope.get("source_url") or "",
                        "proof": proof,
                    },
                    retrieval.chunks,
                    row.question,
                )
                if any(issue.code == "ungrounded_number" for issue in validation.issues):
                    ungrounded_numbers += 1

        leak = _has_pii(envelope)
        pii_leaks += int(leak)
        cases.append(
            {
                "id": row.id,
                "expected_type": row.expected_type,
                "actual_type": actual_type,
                "fact_hit": fact_hit,
                "expected_facts": row.facts,
                "source_hit": source_hit,
                "expected_source_ids": row.source_ids,
                "source_url": envelope.get("source_url"),
                "url_count": len(urls),
                "answer": answer if actual_type == "answer" else None,
                "proof": proof if actual_type == "answer" else None,
                "pii_leak": leak,
            }
        )

    factual_accuracy = factual_correct / factual_total if factual_total else 0.0
    citation_precision = citation_correct / citation_total if citation_total else 0.0
    passed = (
        factual_total > 0
        and factual_accuracy >= 0.90
        and citation_precision == 1.0
        and sentence_violations == 0
        and ungrounded_numbers == 0
        and pii_leaks == 0
    )
    return {
        "suite": "qa",
        "generated_at": _now(),
        "summary": {
            "factual_total": factual_total,
            "factual_correct": factual_correct,
            "factual_accuracy": round(factual_accuracy, 4),
            "factual_threshold": 0.90,
            "citation_total": citation_total,
            "citation_correct": citation_correct,
            "citation_precision": round(citation_precision, 4),
            "sentence_violations": sentence_violations,
            "ungrounded_numbers": ungrounded_numbers,
            "pii_leaks": pii_leaks,
            "pass": passed,
        },
        "cases": cases,
    }


def _load_model(config: Config):
    from src.ingest.embed import load_model

    return load_model(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HDFC MF Facts evaluations.")
    parser.add_argument(
        "--suite",
        choices=("retrieval", "qa", "safety", "all"),
        default="all",
        help="Evaluation suite to run.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD_PATH),
        help="Path to the gold questions CSV.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.safety.cache_answers = False
    rows = load_gold(Path(args.gold))
    out_dir = config.project_root / "data" / "eval"

    results: dict[str, dict[str, Any]] = {}
    model = None
    if args.suite in {"retrieval", "qa", "all"}:
        model = _load_model(config)

    if args.suite in {"retrieval", "all"}:
        results["retrieval"] = run_retrieval(config, rows, model)
        _write_json(out_dir / "retrieval_eval.json", results["retrieval"])
    if args.suite in {"qa", "all"}:
        results["qa"] = run_qa(config, rows, model, _source_url_map(config))
        _write_json(out_dir / "qa_eval.json", results["qa"])
    if args.suite in {"safety", "all"}:
        results["safety"] = run_safety(config, rows)
        _write_json(out_dir / "safety_eval.json", results["safety"])

    for name, result in results.items():
        summary = result["summary"]
        print(f"{name}: {'PASS' if summary['pass'] else 'FAIL'} {json.dumps(summary)}")

    return 0 if all(result["summary"]["pass"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
