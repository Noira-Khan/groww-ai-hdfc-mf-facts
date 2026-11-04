"""JSON logging that never records user questions, answers, or retrieved text."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

FORBIDDEN_FIELD_HINTS = (
    "question",
    "answer",
    "text",
    "proof",
    "query",
    "prompt",
    "response",
    "content",
    "body",
    "api_key",
    "token",
    "secret",
)

ALLOWED_FIELDS = (
    "request_id",
    "route",
    "type",
    "latency_ms",
    "chunk_ids",
    "validator_reasons",
    "provider",
    "model",
    "cache_hit",
    "status",
    "stage",
    "source_id",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def is_forbidden(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in FORBIDDEN_FIELD_HINTS)


def scrub(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if not is_forbidden(key)}


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("hdfc_mf_facts")
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str = "hdfc_mf_facts") -> logging.Logger:
    if not name.startswith("hdfc_mf_facts"):
        name = f"hdfc_mf_facts.{name}"
    logger = logging.getLogger(name)
    configure_logging()
    return logger


def safe_log(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    logger.log(level, event, extra={"fields": scrub(fields)})
