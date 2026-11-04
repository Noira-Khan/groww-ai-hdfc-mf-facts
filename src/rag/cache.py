"""Salted, question-free cache for validated factual answers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from src.config import Config

CACHE_VERSION = 1
DEFAULT_SALT = "hdfc-mf-facts-cache-v1"


def normalize_question(question: str) -> str:
    text = unicodedata.normalize("NFKC", question or "").lower()
    text = re.sub(r"[^\w%₹.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cache_salt() -> str:
    return os.environ.get("CACHE_SALT") or DEFAULT_SALT


def cache_key(question: str, scheme: str | None = None) -> str:
    payload = f"{_cache_salt()}{normalize_question(question)}{scheme or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_dir(config: Config) -> Path:
    return config.app.artifact_dir / "cache" / "answers"


def _cache_path(config: Config, key: str) -> Path:
    return _cache_dir(config) / f"{key}.json"


def get_cached_answer(config: Config, question: str, scheme: str | None = None) -> dict | None:
    if not config.safety.cache_answers:
        return None
    path = _cache_path(config, cache_key(question, scheme))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    envelope = payload.get("envelope")
    return envelope if isinstance(envelope, dict) else None


def put_cached_answer(
    config: Config,
    question: str,
    scheme: str | None,
    envelope: dict,
) -> None:
    if not config.safety.cache_answers:
        return
    if envelope.get("type") != "answer":
        raise ValueError("Only validated factual answer envelopes may be cached.")

    path = _cache_path(config, cache_key(question, scheme))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "envelope": envelope,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clear_cache(config: Config) -> int:
    directory = _cache_dir(config)
    if not directory.exists():
        return 0
    count = 0
    for path in directory.glob("*.json"):
        path.unlink()
        count += 1
    return count
