from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.ingest.embed import embedding_text, run_embed

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class FakeModel:
    def __init__(self, dimension: int = 8, row_count: int | None = None):
        self.dimension = dimension
        self.row_count = row_count
        self.texts: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        self.texts = list(texts)
        count = self.row_count if self.row_count is not None else len(texts)
        return np.arange(count * self.dimension, dtype=np.float32).reshape(count, self.dimension)

    def __getitem__(self, index):
        raise IndexError(index)


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.vector_store.path = tmp_path / "chroma"
    return cfg


def _write_chunks(config, count: int = 3) -> None:
    chunk_dir = config.app.artifact_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    with (chunk_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "chunk_id": f"chunk{index:04d}",
                        "title": f"Scheme page {index}",
                        "section": f"Section {index}",
                        "text": f"Chunk text number {index}.",
                    }
                )
                + "\n"
            )
    (chunk_dir / "chunking_plan.json").write_text(
        json.dumps({"corpus_version": "abc123"}), encoding="utf-8"
    )


def test_run_embed_writes_vectors_ids_and_manifest(config):
    _write_chunks(config, 3)
    result = run_embed(config, model=FakeModel(dimension=8), force=True)

    assert result["reused"] is False
    assert result["summary"]["chunks"] == 3
    assert result["summary"]["dimension"] == 8
    assert result["manifest"]["corpus_version"] == "abc123"

    out = config.app.artifact_dir / "embeddings"
    vectors = np.load(out / "embeddings.npy")
    ids = json.loads((out / "ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "embedding_manifest.json").read_text(encoding="utf-8"))

    assert vectors.shape == (3, 8)
    assert vectors.dtype == np.float32
    assert ids == ["chunk0000", "chunk0001", "chunk0002"]
    assert manifest["chunk_count"] == 3
    assert manifest["dimension"] == 8
    assert manifest["normalize"] is True
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_manifest_hash_is_deterministic(config):
    _write_chunks(config, 2)
    first = run_embed(config, model=FakeModel(dimension=4), force=True)
    second = run_embed(config, model=FakeModel(dimension=4), force=True)

    assert first["manifest"]["revision"]
    assert first["manifest"]["manifest_hash"] == second["manifest"]["manifest_hash"]


def test_run_embed_reuses_matching_manifest(config):
    _write_chunks(config, 2)
    first = run_embed(config, model=FakeModel(dimension=4), force=True)
    assert first["reused"] is False

    second = run_embed(config)
    assert second["reused"] is True
    assert second["summary"]["chunks"] == 2
    assert second["summary"]["dimension"] == 4


def test_run_embed_force_recomputes(config):
    _write_chunks(config, 2)
    run_embed(config, model=FakeModel(dimension=4), force=True)
    result = run_embed(config, model=FakeModel(dimension=5), force=True)
    assert result["reused"] is False
    assert result["summary"]["dimension"] == 5


def test_run_embed_rejects_count_mismatch(config):
    _write_chunks(config, 3)
    with pytest.raises(ValueError, match="does not match chunk count"):
        run_embed(config, model=FakeModel(dimension=8, row_count=2), force=True)


def test_run_embed_requires_chunks(config):
    with pytest.raises(ValueError, match="Run `--stage chunk` first"):
        run_embed(config)


def test_embedding_text_prefixes_title_and_section():
    chunk = {
        "title": "HDFC ELSS Tax Saver Fund scheme page",
        "section": "Scheme Facts / Lock-in",
        "text": "Lock-in: 3 years",
    }
    assert (
        embedding_text(chunk)
        == "HDFC ELSS Tax Saver Fund scheme page | Scheme Facts / Lock-in | Lock-in: 3 years"
    )
    assert embedding_text({"text": "Plain text"}) == "Plain text"


def test_run_embed_encodes_contextualized_text(config):
    _write_chunks(config, 2)
    model = FakeModel(dimension=4)
    run_embed(config, model=model, force=True)
    assert model.texts == [
        "Scheme page 0 | Section 0 | Chunk text number 0.",
        "Scheme page 1 | Section 1 | Chunk text number 1.",
    ]
