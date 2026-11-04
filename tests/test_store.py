from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.ingest.embed import _write_json, _write_vectors
from src.store.chroma import get_collection, run_index

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.vector_store.path = tmp_path / "chroma"
    cfg.embedding.model = "fake-embed"
    cfg.vector_store.rebuild = False
    return cfg


def _write_corpus(config, count: int = 3, dimension: int = 4) -> None:
    chunk_dir = config.app.artifact_dir / "chunks"
    embedding_dir = config.app.artifact_dir / "embeddings"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    ids = []
    with (chunk_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            chunk_id = f"chunk{index:04d}"
            ids.append(chunk_id)
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "source_id": "hdfc_mid_cap_kim" if index % 2 == 0 else "hdfc_elss_kim",
                        "url": "https://www.hdfcfund.com/kim.pdf",
                        "publisher": "HDFC AMC",
                        "doc_type": "kim",
                        "scheme": "mid_cap" if index % 2 == 0 else "elss",
                        "topic": "exit_load",
                        "section": f"Section {index}",
                        "page": index + 1,
                        "as_on_date": "2025-09-30",
                        "text": f"Chunk text number {index}.",
                        "token_count": 8,
                        "text_hash": f"hash{index}",
                    }
                )
                + "\n"
            )

    vectors = np.eye(count, dimension, dtype=np.float32)
    _write_vectors(embedding_dir / "embeddings.npy", vectors)
    _write_json(embedding_dir / "ids.json", ids)
    _write_json(
        embedding_dir / "embedding_manifest.json",
        {
            "model": config.embedding.model,
            "dimension": dimension,
            "chunk_count": count,
            "corpus_version": "corpus-test",
        },
    )


def test_run_index_upserts_and_filters(config):
    _write_corpus(config, count=3, dimension=4)
    summary = run_index(config)

    assert summary["count"] == 3
    assert summary["dimension"] == 4
    assert summary["rebuilt"] is False

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(config.vector_store.path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(config.vector_store.collection)
    result = collection.query(
        query_embeddings=[[1.0, 0.0, 0.0, 0.0]],
        n_results=3,
        where={"scheme": "mid_cap"},
    )
    assert len(result["ids"][0]) == 2
    by_scheme = collection.query(
        query_embeddings=[[1.0, 0.0, 0.0, 0.0]],
        n_results=3,
        where={"scheme": "elss"},
    )
    assert len(by_scheme["ids"][0]) == 1
    by_publisher = collection.query(
        query_embeddings=[[1.0, 0.0, 0.0, 0.0]],
        n_results=3,
        where={"publisher": "HDFC AMC"},
    )
    assert len(by_publisher["ids"][0]) == 3
    assert set(result["metadatas"][0][0]) >= {
        "chunk_id",
        "source_id",
        "url",
        "publisher",
        "doc_type",
        "scheme",
        "topic",
        "as_on_date",
        "section",
        "page",
        "text_hash",
    }
    assert collection.metadata["embedding_model"] == "fake-embed"
    assert collection.metadata["dimension"] == 4
    assert collection.metadata["corpus_version"] == "corpus-test"
    assert collection.metadata["hnsw:space"] == "cosine"

    loaded = get_collection(config)
    assert loaded.count() == 3


def test_run_index_is_idempotent(config):
    _write_corpus(config, count=3)
    first = run_index(config)
    second = run_index(config)
    assert first["count"] == second["count"] == 3


def test_run_index_rebuild_keeps_count(config):
    _write_corpus(config, count=3)
    first = run_index(config, force=True)
    second = run_index(config, force=True)
    assert first["count"] == second["count"] == 3
    assert second["rebuilt"] is True


def test_run_index_fails_on_model_mismatch(config):
    _write_corpus(config, count=2)
    config.embedding.model = "other-model"
    with pytest.raises(ValueError, match="embedding model"):
        run_index(config)


def test_run_index_requires_embeddings(config):
    with pytest.raises(ValueError, match="Run `--stage"):
        run_index(config)


def test_get_collection_fails_closed_on_model_mismatch(config):
    _write_corpus(config, count=2)
    run_index(config, force=True)
    config.embedding.model = "other-model"

    with pytest.raises(ValueError, match="does not match config"):
        get_collection(config)


def test_get_collection_fails_closed_on_dimension_mismatch(config):
    _write_corpus(config, count=2, dimension=4)
    run_index(config, force=True)

    path = config.app.artifact_dir / "embeddings" / "embedding_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dimension"] = 8
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Collection dimension"):
        get_collection(config)


def test_get_collection_fails_closed_on_corpus_version_mismatch(config):
    _write_corpus(config, count=2)
    run_index(config, force=True)

    path = config.app.artifact_dir / "embeddings" / "embedding_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["corpus_version"] = "new-corpus"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Collection corpus version"):
        get_collection(config)


def test_run_index_rejects_id_order_mismatch(config):
    _write_corpus(config, count=2)
    path = config.app.artifact_dir / "embeddings" / "ids.json"
    ids = json.loads(path.read_text(encoding="utf-8"))
    ids.reverse()
    path.write_text(json.dumps(ids), encoding="utf-8")

    with pytest.raises(ValueError, match="ids do not match chunk ids"):
        run_index(config)
