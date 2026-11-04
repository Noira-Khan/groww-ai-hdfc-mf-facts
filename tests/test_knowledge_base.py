from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.services import knowledge_base as kb
from src.services.kb_state import artifacts_present

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    cfg.vector_store.path = tmp_path / "chroma"
    return cfg


def _touch_artifacts(config, chunks=True, embeddings=True):
    artifacts = config.app.artifact_dir
    if chunks:
        path = artifacts / "chunks" / "chunks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    if embeddings:
        embeddings_dir = artifacts / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        for name in ("ids.json", "embeddings.npy", "embedding_manifest.json"):
            (embeddings_dir / name).write_text("{}", encoding="utf-8")


def test_existing_collection_is_reused(config, monkeypatch):
    calls = []
    monkeypatch.setattr(kb, "collection_ready", lambda cfg: True)
    monkeypatch.setattr(kb, "run_index", lambda *a, **k: calls.append("index"))
    monkeypatch.setattr(kb, "run_load", lambda *a, **k: calls.append("load"))

    kb.ensure_knowledge_base(config, model=object())

    assert calls == []


def test_missing_corpus_runs_all_stages(config, monkeypatch):
    calls = []
    states = iter([False, True])
    monkeypatch.setattr(kb, "collection_ready", lambda cfg: next(states))
    monkeypatch.setattr(kb, "run_load", lambda *a, **k: calls.append("load"))
    monkeypatch.setattr(kb, "run_normalize", lambda *a, **k: calls.append("normalize"))
    monkeypatch.setattr(kb, "run_chunk", lambda *a, **k: calls.append("chunk"))
    monkeypatch.setattr(kb, "run_embed", lambda *a, **k: calls.append("embed"))
    monkeypatch.setattr(kb, "run_index", lambda *a, **k: calls.append("index"))

    kb.ensure_knowledge_base(config, model=object())

    assert calls == ["load", "normalize", "chunk", "embed", "index"]


def test_missing_embeddings_runs_embed_and_index(config, monkeypatch):
    _touch_artifacts(config, chunks=True, embeddings=False)
    calls = []
    states = iter([False, True])
    monkeypatch.setattr(kb, "collection_ready", lambda cfg: next(states))
    monkeypatch.setattr(kb, "run_load", lambda *a, **k: calls.append("load"))
    monkeypatch.setattr(kb, "run_normalize", lambda *a, **k: calls.append("normalize"))
    monkeypatch.setattr(kb, "run_chunk", lambda *a, **k: calls.append("chunk"))
    monkeypatch.setattr(kb, "run_embed", lambda *a, **k: calls.append("embed"))
    monkeypatch.setattr(kb, "run_index", lambda *a, **k: calls.append("index"))

    kb.ensure_knowledge_base(config, model=object())

    assert calls == ["embed", "index"]


def test_missing_index_only_runs_index(config, monkeypatch):
    _touch_artifacts(config, chunks=True, embeddings=True)
    calls = []
    states = iter([False, True])
    monkeypatch.setattr(kb, "collection_ready", lambda cfg: next(states))
    monkeypatch.setattr(kb, "run_load", lambda *a, **k: calls.append("load"))
    monkeypatch.setattr(kb, "run_embed", lambda *a, **k: calls.append("embed"))
    monkeypatch.setattr(kb, "run_index", lambda *a, **k: calls.append("index"))

    kb.ensure_knowledge_base(config, model=object())

    assert calls == ["index"]


def test_source_download_failure_is_wrapped(config, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(kb, "collection_ready", lambda cfg: False)
    monkeypatch.setattr(kb, "run_load", fail)

    with pytest.raises(kb.KnowledgeBaseError, match="Source download failed"):
        kb.ensure_knowledge_base(config, model=object())


def test_index_failure_retries_force_then_wraps(config, monkeypatch):
    _touch_artifacts(config, chunks=True, embeddings=True)
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs.get("force"))
        raise RuntimeError("bad index")

    monkeypatch.setattr(kb, "collection_ready", lambda cfg: False)
    monkeypatch.setattr(kb, "run_index", fail)

    with pytest.raises(kb.KnowledgeBaseError, match="Building the Chroma index failed"):
        kb.ensure_knowledge_base(config, model=object())

    assert calls == [False, True]


def test_progress_callback_reports_all_stages(config, monkeypatch):
    events = []
    states = iter([False, True])
    monkeypatch.setattr(kb, "collection_ready", lambda cfg: next(states))
    monkeypatch.setattr(kb, "run_load", lambda *a, **k: None)
    monkeypatch.setattr(kb, "run_normalize", lambda *a, **k: None)
    monkeypatch.setattr(kb, "run_chunk", lambda *a, **k: None)
    monkeypatch.setattr(kb, "run_embed", lambda *a, **k: None)
    monkeypatch.setattr(kb, "run_index", lambda *a, **k: None)

    kb.ensure_knowledge_base(
        config,
        model=object(),
        progress=lambda stage, fraction, message: events.append(
            (stage, fraction, message)
        ),
    )

    assert {stage for stage, _, _ in events} == {
        "download",
        "normalize",
        "chunk",
        "embed",
        "index",
        "validate",
    }
    fractions = [fraction for _, fraction, _ in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_download_progress_reports_source_count(config, monkeypatch):
    events = []

    def fake_load(*args, **kwargs):
        callback = kwargs.get("on_progress")
        for completed in range(1, 20):
            callback(completed, 19, object())

    states = iter([False, True])
    monkeypatch.setattr(kb, "collection_ready", lambda cfg: next(states))
    monkeypatch.setattr(kb, "run_load", fake_load)
    monkeypatch.setattr(kb, "run_normalize", lambda *a, **k: None)
    monkeypatch.setattr(kb, "run_chunk", lambda *a, **k: None)
    monkeypatch.setattr(kb, "run_embed", lambda *a, **k: None)
    monkeypatch.setattr(kb, "run_index", lambda *a, **k: None)

    kb.ensure_knowledge_base(
        config,
        model=object(),
        progress=lambda stage, fraction, message: events.append(
            (stage, fraction, message)
        ),
    )

    messages = [message for stage, _, message in events if stage == "download"]
    assert any("Downloading official documents (19/19)" in message for message in messages)


def test_artifacts_present_requires_manifest_and_index(config):
    assert artifacts_present(config) is False

    manifest = config.app.artifact_dir / "embeddings" / "embedding_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    assert artifacts_present(config) is False

    summary = config.vector_store.path / "_index_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("{}", encoding="utf-8")
    assert artifacts_present(config) is True


def test_force_rebuilds_even_when_collection_is_ready(config, monkeypatch):
    _touch_artifacts(config, chunks=True, embeddings=True)
    calls = []
    monkeypatch.setattr(kb, "collection_ready", lambda cfg: True)
    monkeypatch.setattr(
        kb,
        "run_index",
        lambda *a, **k: calls.append(k.get("force")),
    )

    kb.ensure_knowledge_base(config, model=object(), force=True)

    assert calls == [True]
