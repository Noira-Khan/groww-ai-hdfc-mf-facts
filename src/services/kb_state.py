"""Cheap knowledge-base readiness checks.

This module intentionally avoids importing ChromaDB or sentence-transformers so
the Streamlit shell can render before any heavy initialization happens.
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config


def _present(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def artifacts_present(config: Config) -> bool:
    """Return True when the generated corpus and index files look present.

    This is a fast file check, not a replacement for the full collection
    validation performed by ``ensure_knowledge_base``.
    """
    artifacts = config.app.artifact_dir
    manifest = artifacts / "embeddings" / "embedding_manifest.json"
    if not _present(manifest):
        return False

    index_dir = config.vector_store.path
    return _present(index_dir / "_index_summary.json") or _present(
        index_dir / "chroma.sqlite3"
    )
