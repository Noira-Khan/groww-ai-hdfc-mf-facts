"""Safe, idempotent knowledge-base initialization.

The generated corpus and Chroma index are not committed. Both frontends can
call ``ensure_knowledge_base`` to open an existing index or build only the
missing artifacts.
"""

from __future__ import annotations

from typing import Any

from src.config import Config
from src.ingest.chunk import run_chunk
from src.ingest.embed import run_embed
from src.ingest.loader import run_load
from src.ingest.normalize import run_normalize
from src.store.chroma import get_collection, run_index


class KnowledgeBaseError(RuntimeError):
    """Raised when the corpus or Chroma index cannot be initialized safely."""


def collection_ready(config: Config) -> bool:
    """Return True when the configured collection opens and validates."""
    try:
        get_collection(config)
        return True
    except Exception:
        return False


def _present(path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _build_missing_corpus(config: Config, model: Any) -> None:
    artifacts = config.app.artifact_dir
    chunks_path = artifacts / "chunks" / "chunks.jsonl"
    ids_path = artifacts / "embeddings" / "ids.json"
    vectors_path = artifacts / "embeddings" / "embeddings.npy"
    manifest_path = artifacts / "embeddings" / "embedding_manifest.json"

    if not _present(chunks_path):
        try:
            run_load(config, inventory_path="sources.csv")
        except Exception as exc:
            raise KnowledgeBaseError(
                "Source download failed. The assistant needs network access to "
                f"the official HDFC AMC, SEBI, and AMFI documents. ({exc})"
            ) from exc
        try:
            run_normalize(config, inventory_path="sources.csv")
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Document normalization failed. ({exc})"
            ) from exc
        try:
            run_chunk(config)
        except Exception as exc:
            raise KnowledgeBaseError(f"Chunking failed. ({exc})") from exc

    embeddings_missing = not (
        _present(ids_path) and _present(vectors_path) and _present(manifest_path)
    )
    if embeddings_missing:
        try:
            run_embed(config, model=model)
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Embedding the corpus failed. ({exc})"
            ) from exc

    try:
        run_index(config, force=config.vector_store.rebuild)
    except Exception as first_error:
        try:
            run_index(config, force=True)
        except Exception as exc:
            raise KnowledgeBaseError(
                "Building the Chroma index failed. "
                f"({exc or first_error})"
            ) from exc

    if not collection_ready(config):
        raise KnowledgeBaseError(
            "The Chroma index was built but could not be opened. "
            "Re-run the ingestion pipeline or clear the Streamlit cache."
        )


def ensure_knowledge_base(config: Config, model: Any = None) -> None:
    """Open an existing index, or build only the missing corpus/index parts."""
    if collection_ready(config):
        return
    _build_missing_corpus(config, model)
