"""Persist chunk embeddings in a local ChromaDB collection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import Config

BATCH_SIZE = 128


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_chunks(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _client(config: Config):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(config.vector_store.path),
        settings=Settings(anonymized_telemetry=False),
    )


def _read_embedding_manifest(config: Config) -> dict:
    path = config.app.artifact_dir / "embeddings" / "embedding_manifest.json"
    if not path.exists():
        raise ValueError(f"Missing {path.name}. Run `--stage embed` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def get_collection(config: Config):
    """Open the configured collection, failing closed on any corpus mismatch."""
    client = _client(config)
    try:
        collection = client.get_collection(config.vector_store.collection)
    except Exception as exc:
        raise ValueError(
            f"Collection {config.vector_store.collection!r} not found. Run `--stage index` first."
        ) from exc

    metadata = collection.metadata or {}
    manifest = _read_embedding_manifest(config)

    expected_model = manifest.get("model")
    if expected_model != config.embedding.model:
        raise ValueError(
            f"Embedding manifest model {expected_model!r} does not match "
            f"config {config.embedding.model!r}. Re-run embed and index."
        )
    if metadata.get("embedding_model") != expected_model:
        raise ValueError(
            f"Collection embedding model {metadata.get('embedding_model')!r} does not match "
            f"manifest {expected_model!r}. Re-run index."
        )

    expected_dimension = manifest.get("dimension")
    if expected_dimension and metadata.get("dimension") != expected_dimension:
        raise ValueError(
            f"Collection dimension {metadata.get('dimension')!r} does not match "
            f"manifest {expected_dimension!r}. Re-run index."
        )

    expected_version = manifest.get("corpus_version")
    if expected_version and metadata.get("corpus_version") != expected_version:
        raise ValueError(
            f"Collection corpus version {metadata.get('corpus_version')!r} does not match "
            f"manifest {expected_version!r}. Re-run index or use --rebuild."
        )

    expected_distance = config.vector_store.distance
    if metadata.get("hnsw:space", expected_distance) != expected_distance:
        raise ValueError(
            f"Collection distance {metadata.get('hnsw:space')!r} does not match "
            f"config {expected_distance!r}. Use --rebuild."
        )
    return collection


def _metadata(chunk: dict) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "url": chunk["url"],
        "publisher": chunk["publisher"],
        "doc_type": chunk["doc_type"],
        "scheme": chunk["scheme"],
        "topic": chunk["topic"],
        "as_on_date": chunk.get("as_on_date") or "",
        "section": chunk.get("section") or "",
        "page": int(chunk["page"]) if chunk.get("page") is not None else -1,
        "text_hash": chunk["text_hash"],
    }


def run_index(config: Config, force: bool = False) -> dict:
    artifact_dir = config.app.artifact_dir
    chunks_path = artifact_dir / "chunks" / "chunks.jsonl"
    ids_path = artifact_dir / "embeddings" / "ids.json"
    vectors_path = artifact_dir / "embeddings" / "embeddings.npy"
    manifest_path = artifact_dir / "embeddings" / "embedding_manifest.json"

    for path, hint in (
        (chunks_path, "Run `--stage chunk` first."),
        (ids_path, "Run `--stage embed` first."),
        (vectors_path, "Run `--stage embed` first."),
        (manifest_path, "Run `--stage embed` first."),
    ):
        if not path.exists():
            raise ValueError(f"Missing {path.name}. {hint}")

    chunks = _read_chunks(chunks_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
    embedding_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not chunks or not ids:
        raise ValueError("Chunks or ids are empty. Re-run chunk and embed.")
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2-dimensional, got shape {vectors.shape}")
    if len(chunks) != len(ids) or vectors.shape[0] != len(ids):
        raise ValueError(
            f"Count mismatch: chunks={len(chunks)}, ids={len(ids)}, vectors={vectors.shape[0]}"
        )
    expected_ids = [chunk["chunk_id"] for chunk in chunks]
    if ids != expected_ids:
        raise ValueError(
            "Embedding ids do not match chunk ids or order. Re-run `--stage embed`."
        )
    manifest_count = embedding_manifest.get("chunk_count")
    if manifest_count is not None and manifest_count != len(chunks):
        raise ValueError(
            f"Embedding manifest chunk count {manifest_count} does not match "
            f"chunk count {len(chunks)}. Re-run `--stage embed`."
        )

    expected_dimension = int(embedding_manifest.get("dimension") or 0)
    if expected_dimension and vectors.shape[1] != expected_dimension:
        raise ValueError(
            f"Embedding dimension {vectors.shape[1]} does not match manifest {expected_dimension}"
        )
    if config.embedding.model != embedding_manifest.get("model"):
        raise ValueError(
            "Configured embedding model does not match embedding manifest: "
            f"{config.embedding.model!r} != {embedding_manifest.get('model')!r}"
        )

    corpus_version = embedding_manifest.get("corpus_version", "unknown")
    client = _client(config)
    name = config.vector_store.collection
    rebuild = force or config.vector_store.rebuild

    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    metadata = {
        "hnsw:space": config.vector_store.distance,
        "embedding_model": config.embedding.model,
        "dimension": int(vectors.shape[1]),
        "corpus_version": corpus_version,
        "chunk_count": len(chunks),
        "ingested_at": _now(),
    }

    try:
        collection = client.get_collection(name)
    except Exception:
        collection = None

    if collection is not None:
        existing = collection.metadata or {}
        for key in ("embedding_model", "dimension"):
            if key in existing and existing[key] != metadata[key]:
                raise ValueError(
                    f"Collection {name} has {key}={existing[key]!r}, expected {metadata[key]!r}. "
                    "Use --rebuild to recreate the collection."
                )
        existing_version = existing.get("corpus_version")
        if not rebuild and existing_version and existing_version != corpus_version:
            raise ValueError(
                f"Collection {name} has corpus_version={existing_version!r}, expected "
                f"{corpus_version!r}. Use --rebuild to recreate the collection."
            )
        if existing.get("hnsw:space", config.vector_store.distance) != config.vector_store.distance:
            raise ValueError(
                f"Collection {name} uses distance {existing.get('hnsw:space')!r}, "
                f"expected {config.vector_store.distance!r}. Use --rebuild."
            )
        merged = {**existing, **{key: value for key, value in metadata.items() if key != "hnsw:space"}}
        merged.pop("hnsw:space", None)
        collection.modify(metadata=merged)
    else:
        collection = client.create_collection(
            name=name,
            metadata=metadata,
        )

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [_metadata(chunk) for chunk in chunks]
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    count = collection.count()
    if count != len(chunks):
        raise ValueError(f"Collection count {count} does not match chunk count {len(chunks)}")

    summary = {
        "collection": name,
        "path": str(config.vector_store.path),
        "count": count,
        "distance": config.vector_store.distance,
        "embedding_model": config.embedding.model,
        "dimension": int(vectors.shape[1]),
        "corpus_version": corpus_version,
        "rebuilt": rebuild,
        "ingested_at": metadata["ingested_at"],
    }
    _write_json(artifact_dir / "chroma" / "_index_summary.json", summary)
    return summary
