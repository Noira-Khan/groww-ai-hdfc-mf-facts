"""Configuration loading and validation.

All pipeline parameters live in config.yaml. Secrets live only in the
environment (.env locally) and are never stored in the config object or logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

PROVIDER_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

SECRET_HINTS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "secret",
    "password",
    "salt",
)

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    disclaimer: str
    max_answer_sentences: int = Field(default=3, ge=1, le=10)
    artifact_dir: Path


class IngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowlist_publishers: list[str] = Field(min_length=1)
    request_timeout_s: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=1, le=10)
    request_delay_ms: int = Field(default=800, ge=0, le=10000)
    user_agent: str = "Mozilla/5.0 (compatible; HDFC-MF-Facts-Prototype/1.0)"


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner: Literal["auto", "section_aware", "recursive", "fixed", "llm"] = "auto"
    target_tokens: int = Field(default=800, ge=100, le=4000)
    overlap_tokens: int = Field(default=120, ge=0, le=1000)
    keep_tables_intact: bool = True


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=512)
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    max_seq_tokens: int = Field(default=256, ge=64, le=8192)


class VectorStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["chroma"] = "chroma"
    path: Path
    collection: str = Field(min_length=1)
    distance: Literal["cosine", "l2"] = "cosine"
    rebuild: bool = False


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_context_chunks: int = Field(default=3, ge=1, le=10)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "groq"
    model: str = ""
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=400, ge=1, le=8192)
    timeout_s: int = Field(default=30, ge=5, le=180)


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detect_pii: bool = True
    max_repair_retries: int = Field(default=1, ge=0, le=3)
    cache_answers: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    app: AppConfig
    ingestion: IngestionConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    safety: SafetyConfig
    project_root: Path = Field(default=Path("."), exclude=True)

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else (self.project_root / value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1) or m.group(2), ""), value)
    return value


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Load config.yaml, expand ${ENV_VARS}, and resolve paths."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    env_file = path.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")

    expanded = _expand_env(raw)
    config = Config(**expanded)
    config.project_root = path.parent
    config.app.artifact_dir = config.resolve_path(config.app.artifact_dir)
    config.vector_store.path = config.resolve_path(config.vector_store.path)
    return config


def ensure_artifact_dirs(config: Config) -> None:
    artifact_dir = config.app.artifact_dir
    for name in ("raw", "normalized", "chunks", "embeddings", "chroma", "eval"):
        (artifact_dir / name).mkdir(parents=True, exist_ok=True)


def validate_runtime(config: Config) -> list[str]:
    """Structural checks are enforced by Pydantic. This returns environment warnings."""
    warnings: list[str] = []

    if config.llm.provider not in PROVIDER_KEY_ENV:
        raise ValueError(
            f"Unknown LLM provider '{config.llm.provider}'. "
            f"Choose one of: {', '.join(sorted(PROVIDER_KEY_ENV))}"
        )

    if not config.llm.model.strip():
        warnings.append("LLM_MODEL_ID is empty. Set it in .env before calling the model.")

    key_env = PROVIDER_KEY_ENV[config.llm.provider]
    if not os.environ.get(key_env):
        warnings.append(f"{key_env} is not set. The configured provider will fail at request time.")

    if config.chunking.overlap_tokens >= config.chunking.target_tokens:
        raise ValueError("chunking.overlap_tokens must be smaller than chunking.target_tokens")

    if config.retrieval.max_context_chunks > config.retrieval.top_k:
        raise ValueError("retrieval.max_context_chunks cannot exceed retrieval.top_k")

    if not str(config.vector_store.collection).strip():
        raise ValueError("vector_store.collection must not be empty")

    ensure_artifact_dirs(config)
    return warnings


def _mask(value: Any, key: str = "") -> Any:
    if key and any(hint in key.lower() for hint in SECRET_HINTS):
        return "***set***" if value else "***missing***"
    if isinstance(value, dict):
        return {item_key: _mask(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_mask(item) for item in value]
    return value


def masked_dict(config: Config) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    data = _mask(data)
    data["llm"]["model"] = config.llm.model or "***missing***"
    data["llm"]["provider_key_env"] = PROVIDER_KEY_ENV[config.llm.provider]
    data["provider_key_present"] = bool(os.environ.get(PROVIDER_KEY_ENV[config.llm.provider]))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load and validate the assistant config.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        warnings = validate_runtime(config)
    except Exception as exc:  # ConfigError is intentionally not imported here.
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(masked_dict(config), indent=2))
    if warnings:
        print("\nWARNINGS:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
        return 1 if args.strict else 0

    print("\nConfig OK. No warnings.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
