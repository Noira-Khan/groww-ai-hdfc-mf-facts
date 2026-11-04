"""Embed chunks with the configured sentence-transformers model."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import Config

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONTEXT_HEADER_VERSION = "context_header_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunks.append(json.loads(line))
    return chunks


def _model_revision(model, model_name: str) -> str:
    try:
        first = model[0]
        config = first.auto_model.config
        commit = getattr(config, "_commit_hash", None)
        if commit:
            return commit
    except Exception:
        pass
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str):
            return Path(cached).parent.name
    except Exception:
        pass
    return "unknown"


def _manifest_hash(manifest: dict) -> str:
    stable = {
        key: manifest.get(key)
        for key in (
            "model",
            "revision",
            "dimension",
            "chunk_count",
            "chunk_ids_sha256",
            "corpus_version",
            "normalize",
            "batch_size",
            "max_seq_tokens",
            "context_header",
            "dtype",
        )
    }
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embedding_text(chunk: dict) -> str:
    """Prepend source context so short fact chunks remain retrievable."""
    parts = [chunk.get("title") or ""]
    section = chunk.get("section") or ""
    if section and section != "Document":
        parts.append(section)
    parts.append(chunk.get("text") or "")
    return " | ".join(part.strip() for part in parts if part and part.strip())


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_vectors(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, vectors.astype(np.float32))
    tmp.replace(path)


def load_model(config: Config):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        config.embedding.model or DEFAULT_MODEL,
        device=config.embedding.device,
    )


def run_embed(
    config: Config,
    chunks_path: str | Path | None = None,
    force: bool = False,
    model=None,
) -> dict:
    path = (
        Path(chunks_path)
        if chunks_path is not None
        else config.app.artifact_dir / "chunks" / "chunks.jsonl"
    )
    if not path.is_absolute():
        path = config.project_root / path
    if not path.exists():
        raise ValueError("No chunks file found. Run `--stage chunk` first.")

    chunks = _load_chunks(path)
    if not chunks:
        raise ValueError("Chunks file is empty. Run `--stage chunk` first.")

    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [embedding_text(chunk) for chunk in chunks]
    plan_path = path.parent / "chunking_plan.json"
    corpus_version = "unknown"
    if plan_path.exists():
        corpus_version = json.loads(plan_path.read_text(encoding="utf-8")).get(
            "corpus_version", "unknown"
        )
    ids_hash = _sha256(ids)
    out_dir = config.app.artifact_dir / "embeddings"
    vectors_path = out_dir / "embeddings.npy"
    ids_path = out_dir / "ids.json"
    manifest_path = out_dir / "embedding_manifest.json"

    if not force and vectors_path.exists() and ids_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("model") == config.embedding.model
            and existing.get("chunk_count") == len(ids)
            and existing.get("chunk_ids_sha256") == ids_hash
            and existing.get("corpus_version") == corpus_version
            and existing.get("context_header") == CONTEXT_HEADER_VERSION
            and existing.get("manifest_hash") == _manifest_hash(existing)
        ):
            return {
                "reused": True,
                "summary": {
                    "chunks": len(ids),
                    "dimension": existing.get("dimension"),
                    "model": existing.get("model"),
                },
                "paths": {
                    "vectors": str(vectors_path),
                    "ids": str(ids_path),
                    "manifest": str(manifest_path),
                },
            }

    if model is None:
        model = load_model(config)

    vectors = model.encode(
        texts,
        batch_size=config.embedding.batch_size,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Embedding output must be 2-dimensional, got shape {vectors.shape}")
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding count {vectors.shape[0]} does not match chunk count {len(chunks)}"
        )
    if config.embedding.normalize:
        vectors = _normalize_rows(vectors)

    dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    expected_dimension = dimension_getter() if dimension_getter else None
    if expected_dimension and vectors.shape[1] != expected_dimension:
        raise ValueError(
            f"Embedding dimension {vectors.shape[1]} does not match model dimension "
            f"{expected_dimension}"
        )

    _write_vectors(vectors_path, vectors)
    _write_json(ids_path, ids)

    manifest = {
        "created_at": _now(),
        "model": config.embedding.model,
        "revision": _model_revision(model, config.embedding.model),
        "dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "chunk_ids_sha256": ids_hash,
        "corpus_version": corpus_version,
        "normalize": config.embedding.normalize,
        "batch_size": config.embedding.batch_size,
        "device": config.embedding.device,
        "max_seq_tokens": config.embedding.max_seq_tokens,
        "context_header": CONTEXT_HEADER_VERSION,
        "dtype": str(vectors.dtype),
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    _write_json(manifest_path, manifest)

    return {
        "reused": False,
        "summary": {
            "chunks": len(chunks),
            "dimension": int(vectors.shape[1]),
            "model": config.embedding.model,
        },
        "paths": {
            "vectors": str(vectors_path),
            "ids": str(ids_path),
            "manifest": str(manifest_path),
        },
        "manifest": manifest,
    }
