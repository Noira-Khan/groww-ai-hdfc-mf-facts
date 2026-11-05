from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    _mask,
    ensure_artifact_dirs,
    load_config,
    masked_dict,
    validate_runtime,
)
from src.logging import scrub

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def config():
    cfg = load_config(CONFIG_PATH)
    return cfg


def test_loads_all_sections(config):
    assert config.app.name == "Groww Fund Facts"
    assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.vector_store.provider == "chroma"
    assert config.retrieval.max_context_chunks <= config.retrieval.top_k
    assert config.project_root == CONFIG_PATH.parent


def test_env_var_expansion(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_ID", "test-model-id")
    cfg = load_config(CONFIG_PATH)
    assert cfg.llm.model == "test-model-id"


def test_paths_resolved_under_project_root(config):
    assert config.app.artifact_dir.is_absolute()
    assert config.vector_store.path.is_absolute()
    assert config.app.artifact_dir.name == "artifacts"


def test_unknown_provider_rejected(config):
    config.llm.provider = "not-a-provider"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        validate_runtime(config)


def test_overlap_must_be_smaller_than_target(config):
    config.chunking.overlap_tokens = config.chunking.target_tokens
    with pytest.raises(ValueError, match="overlap_tokens"):
        validate_runtime(config)


def test_context_chunks_cannot_exceed_top_k(config):
    config.retrieval.max_context_chunks = config.retrieval.top_k + 1
    with pytest.raises(ValueError, match="max_context_chunks"):
        validate_runtime(config)


def test_missing_env_is_a_warning_not_a_crash(config, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL_ID", "test-model-id")
    warnings = validate_runtime(config)
    assert any("GROQ_API_KEY" in warning for warning in warnings)


def test_artifact_dirs_created(config, tmp_path):
    config.app.artifact_dir = tmp_path / "artifacts"
    ensure_artifact_dirs(config)
    for name in ("raw", "normalized", "chunks", "embeddings", "chroma", "eval"):
        assert (config.app.artifact_dir / name).is_dir()


def test_masking_hides_secrets():
    masked = _mask({"api_key": "abc", "normal": "value"})
    assert masked == {"api_key": "***set***", "normal": "value"}


def test_masked_config_never_contains_provider_key(config):
    rendered = masked_dict(config)
    assert "provider_key_env" in rendered["llm"]
    assert rendered["llm"]["model"] == (config.llm.model or "***missing***")


def test_log_scrub_drops_sensitive_fields():
    fields = {"question": "my PAN is ...", "answer": "x", "request_id": "r1", "route": "/api/chat"}
    assert scrub(fields) == {"request_id": "r1", "route": "/api/chat"}
