"""FastAPI application for the HDFC MF Facts assistant."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.config import Config, load_config, masked_dict, validate_runtime
from src.logging import get_logger, safe_log
from src.rag.pipeline import answer_question
from src.safety import responses
from src.safety.intent import SCHEME_LABELS, classify_question
from src.safety.pii import detect_pii
from src.store.chroma import get_collection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = PROJECT_ROOT / "ui"
PIPELINE_STAGES = ("load", "chunk", "embed", "store", "retrieve")

logger = get_logger("api")


@lru_cache(maxsize=2)
def _load_sentence_model(model: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model, device=device)


def get_model(config: Config):
    return _load_sentence_model(config.embedding.model, config.embedding.device)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    scheme: Literal["large_cap", "flexi_cap", "elss", "mid_cap", "small_cap"] | None = None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_chunks(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


def _source_rows(config: Config) -> list[dict]:
    inventory_path = config.project_root / "sources.csv"
    rows: list[dict] = []
    if inventory_path.exists():
        with inventory_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "source_id": row.get("source_id", ""),
                        "url": row.get("url", ""),
                        "publisher": row.get("publisher", ""),
                        "doc_type": row.get("doc_type", ""),
                        "scheme": row.get("scheme", ""),
                        "title": row.get("title", ""),
                        "expected_topics": [
                            topic
                            for topic in (row.get("expected_topics") or "").split("|")
                            if topic
                        ],
                    }
                )
    return rows


def _source_inventory(config: Config) -> list[dict]:
    rows = _source_rows(config)
    manifest = _read_json(config.app.artifact_dir / "raw" / "manifest.json") or {}
    stats = _read_json(config.app.artifact_dir / "chunks" / "chunk_stats.json") or {}
    by_source = stats.get("by_source", {})
    manifest_by_id = {
        entry.get("source_id"): entry
        for entry in manifest.get("sources", [])
        if isinstance(entry, dict)
    }

    if not rows:
        rows = [
            {
                "source_id": entry.get("source_id", ""),
                "url": entry.get("url", ""),
                "publisher": entry.get("publisher", ""),
                "doc_type": entry.get("doc_type", ""),
                "scheme": entry.get("scheme", ""),
                "title": entry.get("title", ""),
                "expected_topics": [],
            }
            for entry in manifest.get("sources", [])
            if isinstance(entry, dict)
        ]

    merged: list[dict] = []
    for row in rows:
        entry = manifest_by_id.get(row["source_id"], {})
        merged.append(
            {
                **row,
                "status": entry.get("status", "not_ingested"),
                "http_status": entry.get("http_status"),
                "fetched_at": entry.get("fetched_at"),
                "sha256": entry.get("sha256"),
                "local_path": entry.get("local_path"),
                "chunks": by_source.get(row["source_id"], {}).get("chunks", 0),
            }
        )
    return merged


def _pipeline_summary(config: Config) -> dict[str, Any]:
    artifact_dir = config.app.artifact_dir
    raw_manifest = _read_json(artifact_dir / "raw" / "manifest.json") or {}
    chunk_stats = _read_json(artifact_dir / "chunks" / "chunk_stats.json") or {}
    chunk_plan = _read_json(artifact_dir / "chunks" / "chunking_plan.json") or {}
    embedding_manifest = _read_json(
        artifact_dir / "embeddings" / "embedding_manifest.json"
    ) or {}
    index_summary = _read_json(artifact_dir / "chroma" / "_index_summary.json") or {}
    retrieval_dir = artifact_dir / "retrieval"
    traces = len(list(retrieval_dir.glob("*.json"))) if retrieval_dir.exists() else 0

    sample_chunks = []
    for chunk in _read_chunks(artifact_dir / "chunks" / "chunks.jsonl", limit=5):
        sample_chunks.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "source_id": chunk.get("source_id"),
                "publisher": chunk.get("publisher"),
                "scheme": chunk.get("scheme"),
                "topic": chunk.get("topic"),
                "section": chunk.get("section"),
                "url": chunk.get("url"),
                "text": (chunk.get("text") or "")[:600],
            }
        )

    stages = {
        "load": {
            "generated_at": raw_manifest.get("generated_at"),
            "summary": raw_manifest.get("summary", {}),
            "inventory": raw_manifest.get("inventory"),
            "manifest_path": raw_manifest.get("manifest_path"),
        },
        "chunk": {
            "plan": chunk_plan,
            "stats": chunk_stats,
        },
        "embed": embedding_manifest,
        "store": index_summary,
        "retrieve": {
            "top_k": config.retrieval.top_k,
            "score_threshold": config.retrieval.score_threshold,
            "max_context_chunks": config.retrieval.max_context_chunks,
            "traces": traces,
        },
    }

    return {
        "counts": {
            "sources": raw_manifest.get("summary", {}).get("total", 0),
            "chunks": chunk_stats.get("total_chunks", 0),
            "embeddings": embedding_manifest.get("chunk_count", 0),
            "vectors": index_summary.get("count", 0),
            "retrieval_traces": traces,
        },
        "stages": stages,
        "sample_chunks": sample_chunks,
    }


def _config_hash(config: Config) -> str:
    payload = json.dumps(masked_dict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _health(config: Config) -> dict[str, Any]:
    collection_count = None
    metadata: dict[str, Any] = {}
    collection_error: str | None = None
    try:
        collection = get_collection(config)
        collection_count = collection.count()
        metadata = dict(collection.metadata or {})
    except Exception as exc:
        collection_error = str(exc) or "Collection is not available."

    try:
        warnings = validate_runtime(config)
    except Exception:
        warnings = ["Runtime validation failed."]

    next_step = None
    if collection_error:
        next_step = "Run `python -m src.ingest --stage all` to build the corpus and index."
        warnings.append(next_step)

    status = "degraded" if collection_error else "ok"
    return {
        "status": status,
        "app_name": config.app.name,
        "disclaimer": config.app.disclaimer,
        "provider": config.llm.provider,
        "model": config.llm.model,
        "embedding_model": metadata.get("embedding_model", config.embedding.model),
        "dimension": metadata.get("dimension"),
        "collection": config.vector_store.collection,
        "collection_count": collection_count,
        "corpus_version": metadata.get("corpus_version"),
        "collection_error": collection_error,
        "next_step": next_step,
        "config_hash": _config_hash(config),
        "warnings": warnings,
    }


def _log_chat(
    request_id: str,
    response_type: str,
    route: str,
    started: float,
    config: Config,
    chunks: list[str] | None = None,
) -> None:
    safe_log(
        logger,
        "chat",
        request_id=request_id,
        type=response_type,
        route=route,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        chunk_ids=chunks or [],
        provider=config.llm.provider,
        model=config.llm.model,
    )


def create_app(config: Config | None = None, warmup: bool = True) -> FastAPI:
    cfg = config or load_config(os.environ.get("HDFC_CONFIG", "config.yaml"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        validate_runtime(cfg)
        if warmup:
            try:
                get_model(cfg)
            except Exception:
                safe_log(logger, "model_warmup_failed", stage="embedding")
        yield

    app = FastAPI(title=cfg.app.name, lifespan=lifespan)
    app.state.config = cfg

    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        malformed = any(
            error.get("type") == "json_invalid" for error in exc.errors()
        )
        status = 400 if malformed else 422
        return JSONResponse(
            status_code=status,
            content=responses.error_response(uuid.uuid4().hex[:16]),
        )

    @app.get("/")
    async def index():
        path = UI_DIR / "index.html"
        if not path.exists():
            return JSONResponse(status_code=404, content={"detail": "UI not found"})
        return FileResponse(path)

    @app.post("/api/chat")
    async def chat(payload: ChatRequest, request: Request):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:16]
        question = payload.question.strip()
        if not question:
            return JSONResponse(
                status_code=422,
                content=responses.error_response(request_id),
            )

        route = "answer"
        if cfg.safety.detect_pii and detect_pii(question):
            route = "pii"
            envelope = responses.pii_response(request_id)
        else:
            intent = classify_question(question, payload.scheme)
            route = intent.route
            if intent.route == "refusal_advice":
                envelope = responses.advice_response(request_id)
            elif intent.route == "refusal_performance":
                envelope = responses.performance_response(request_id)
            elif intent.route == "not_in_sources":
                envelope = responses.not_in_sources_response(request_id)
            elif intent.route == "clarify":
                envelope = responses.clarify_response(question, request_id)
            else:
                scheme = intent.first_scheme if intent.scheme_sensitive else None
                try:
                    envelope = answer_question(
                        cfg,
                        question,
                        scheme=scheme,
                        model=get_model(cfg),
                    )
                    if envelope.get("type") == "answer" and intent.extra_schemes:
                        envelope["can_do"] = [
                            f"What about {SCHEME_LABELS[item]}?"
                            for item in intent.extra_schemes
                        ]
                except ValueError:
                    safe_log(
                        logger,
                        "chat_unavailable",
                        request_id=request_id,
                        route=route,
                    )
                    return JSONResponse(
                        status_code=503,
                        content=responses.error_response(request_id),
                    )
                except Exception:
                    envelope = responses.error_response(request_id)

        _log_chat(
            request_id,
            envelope["type"],
            route,
            started,
            cfg,
            envelope.get("chunks_used"),
        )
        return envelope

    @app.get("/api/sources")
    async def sources():
        inventory = _source_inventory(cfg)
        return {"count": len(inventory), "sources": inventory}

    @app.get("/api/pipeline")
    async def pipeline():
        return _pipeline_summary(cfg)

    @app.get("/api/pipeline/{stage}")
    async def pipeline_stage(stage: str):
        if stage not in PIPELINE_STAGES:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Unknown pipeline stage: {stage}"},
            )
        summary = _pipeline_summary(cfg)
        return {"stage": stage, **summary["stages"][stage]}

    @app.get("/health")
    async def health():
        return _health(cfg)

    return app


app = create_app()
