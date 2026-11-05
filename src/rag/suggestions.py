"""Verify and persist confident 'Try these' questions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import Config
from src.rag.retrieve import retrieve
from src.safety.intent import classify_question
from src.suggestions import CURATED_SUGGESTIONS, DEFAULT_SUGGESTIONS, Suggestion

SUGGESTIONS_VERSION = 1


def suggestions_path(config: Config) -> Path:
    return config.app.artifact_dir / "suggestions.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _current_corpus_version(config: Config) -> str | None:
    path = config.app.artifact_dir / "embeddings" / "embedding_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest.get("corpus_version")


def _settings(config: Config) -> dict:
    return {
        "corpus_version": _current_corpus_version(config),
        "topic_boost": config.retrieval.topic_boost,
        "suggestion_score_threshold": config.retrieval.suggestion_score_threshold,
    }


def load_suggestions(config: Config, limit: int = 6) -> list[dict] | None:
    """Return a matching cached suggestion list, or None when stale/missing."""
    path = suggestions_path(config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != SUGGESTIONS_VERSION:
        return None

    current = _current_corpus_version(config)
    if current and payload.get("corpus_version") != current:
        return None
    if payload.get("topic_boost") != config.retrieval.topic_boost:
        return None
    if (
        payload.get("suggestion_score_threshold")
        != config.retrieval.suggestion_score_threshold
    ):
        return None

    items = payload.get("suggestions")
    if not isinstance(items, list):
        return None
    valid = [
        item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("question"), str)
    ]
    return valid[:limit]


def resolve_suggestions(config: Config, limit: int = 6) -> list[str]:
    """Return verified questions, falling back to the curated defaults."""
    cached = load_suggestions(config, limit=limit)
    if cached is not None:
        return [item["question"] for item in cached]
    return list(DEFAULT_SUGGESTIONS[:limit])


def _verify(
    config: Config, suggestion: Suggestion, model
) -> dict | None:
    intent = classify_question(suggestion.question)
    if intent.route != "answer":
        return None

    result = retrieve(
        config,
        suggestion.question,
        scheme=suggestion.scheme,
        model=model,
    )
    if not result.chunks:
        return None

    top = result.chunks[0]
    if suggestion.topic and top.metadata.get("topic") != suggestion.topic:
        return None
    if suggestion.scheme and top.metadata.get("scheme") != suggestion.scheme:
        return None

    required = max(
        suggestion.min_score, config.retrieval.suggestion_score_threshold
    )
    if top.score < required:
        return None

    return {
        "question": suggestion.question,
        "topic": suggestion.topic,
        "scheme": suggestion.scheme,
        "score": round(top.score, 4),
        "chunk_id": top.chunk_id,
        "source_id": top.metadata.get("source_id"),
    }


def build_suggestions(config: Config, model=None, limit: int = 6) -> list[dict]:
    """Retrieve each curated question and keep only confident matches."""
    if model is None:
        from src.ingest.embed import load_model

        model = load_model(config)

    verified: list[dict] = []
    seen: set[str] = set()
    for suggestion in CURATED_SUGGESTIONS:
        if len(verified) >= limit:
            break
        if suggestion.question in seen:
            continue
        item = _verify(config, suggestion, model)
        if item is not None:
            verified.append(item)
            seen.add(suggestion.question)
    return verified


def write_suggestions(config: Config, model=None, limit: int = 6) -> dict:
    """Verify, then write the current suggestion artifact."""
    cached = load_suggestions(config, limit=limit)
    if cached is not None:
        return {
            "reused": True,
            "path": str(suggestions_path(config)),
            "count": len(cached),
            "suggestions": cached,
        }

    items = build_suggestions(config, model=model, limit=limit)
    payload = {
        "version": SUGGESTIONS_VERSION,
        "generated_at": _now(),
        **_settings(config),
        "suggestions": items,
    }
    path = suggestions_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(path)
    return {
        "reused": False,
        "path": str(path),
        "count": len(items),
        "suggestions": items,
    }
