"""Safe, idempotent knowledge-base initialization.

The generated corpus and Chroma index are not committed. Both frontends can
call ``ensure_knowledge_base`` to open an existing index or build only the
missing artifacts. A progress callback lets the Streamlit frontend report
stage-level progress without changing any retrieval or validation behavior.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from src.config import Config
from src.ingest.chunk import run_chunk
from src.ingest.embed import run_embed
from src.ingest.loader import run_load
from src.ingest.normalize import run_normalize
from src.logging import get_logger, safe_log
from src.store.chroma import get_collection, run_index

ProgressCallback = Callable[[str, float, str], None]

logger = get_logger("services.knowledge_base")


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


def _emit(
    progress: ProgressCallback | None, stage: str, fraction: float, message: str
) -> None:
    if progress is None:
        return
    try:
        progress(stage, max(0.0, min(1.0, float(fraction))), message)
    except Exception:
        # Progress reporting must never break initialization.
        return


def _run_stage(
    stage: str,
    start: float,
    end: float,
    start_message: str,
    done_message: str,
    action: Callable[[], Any],
    progress: ProgressCallback | None,
) -> Any:
    started = time.perf_counter()
    _emit(progress, stage, start, start_message)
    try:
        result = action()
    except Exception:
        safe_log(
            logger,
            "kb_stage",
            stage=stage,
            status="failed",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        raise
    safe_log(
        logger,
        "kb_stage",
        stage=stage,
        status="ok",
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    _emit(progress, stage, end, done_message)
    return result


def _build_missing_corpus(
    config: Config,
    model: Any,
    progress: ProgressCallback | None = None,
    force_index: bool = False,
) -> None:
    artifacts = config.app.artifact_dir
    chunks_path = artifacts / "chunks" / "chunks.jsonl"
    ids_path = artifacts / "embeddings" / "ids.json"
    vectors_path = artifacts / "embeddings" / "embeddings.npy"
    manifest_path = artifacts / "embeddings" / "embedding_manifest.json"

    if not _present(chunks_path):

        def download():
            def on_source(completed: int, total: int, result) -> None:
                fraction = 0.05 + 0.40 * (completed / max(1, total))
                _emit(
                    progress,
                    "download",
                    fraction,
                    f"Downloading official documents ({completed}/{total})…",
                )

            return run_load(
                config,
                inventory_path="sources.csv",
                on_progress=on_source,
            )

        try:
            _run_stage(
                "download",
                0.05,
                0.45,
                "Downloading official documents…",
                "Official documents downloaded.",
                download,
                progress,
            )
        except Exception as exc:
            raise KnowledgeBaseError(
                "Source download failed. The assistant needs network access to "
                f"the official HDFC AMC, SEBI, and AMFI documents. ({exc})"
            ) from exc
        try:
            _run_stage(
                "normalize",
                0.45,
                0.55,
                "Normalizing documents…",
                "Documents normalized.",
                lambda: run_normalize(config, inventory_path="sources.csv"),
                progress,
            )
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Document normalization failed. ({exc})"
            ) from exc
        try:
            _run_stage(
                "chunk",
                0.55,
                0.65,
                "Chunking documents…",
                "Documents chunked.",
                lambda: run_chunk(config),
                progress,
            )
        except Exception as exc:
            raise KnowledgeBaseError(f"Chunking failed. ({exc})") from exc
    else:
        _emit(progress, "download", 0.65, "Using existing document chunks.")

    embeddings_missing = not (
        _present(ids_path) and _present(vectors_path) and _present(manifest_path)
    )
    if embeddings_missing:
        try:
            _run_stage(
                "embed",
                0.65,
                0.85,
                "Embedding corpus chunks…",
                "Corpus embedded.",
                lambda: run_embed(config, model=model),
                progress,
            )
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Embedding the corpus failed. ({exc})"
            ) from exc
    else:
        _emit(progress, "embed", 0.85, "Using existing embeddings.")

    def build_index():
        try:
            return run_index(
                config, force=force_index or config.vector_store.rebuild
            )
        except Exception as first_error:
            try:
                return run_index(config, force=True)
            except Exception as exc:
                raise KnowledgeBaseError(
                    "Building the Chroma index failed. "
                    f"({exc or first_error})"
                ) from exc

    try:
        _run_stage(
            "index",
            0.85,
            0.97,
            "Building the Chroma index…",
            "Chroma index built.",
            build_index,
            progress,
        )
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(f"Building the Chroma index failed. ({exc})") from exc

    def validate() -> None:
        if not collection_ready(config):
            raise KnowledgeBaseError(
                "The Chroma index was built but could not be opened. "
                "Re-run the ingestion pipeline or clear the Streamlit cache."
            )

    try:
        _run_stage(
            "validate",
            0.97,
            1.0,
            "Validating the Chroma collection…",
            "Knowledge base ready.",
            validate,
            progress,
        )
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Knowledge-base validation failed. ({exc})"
        ) from exc


def ensure_knowledge_base(
    config: Config,
    model: Any = None,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> None:
    """Open an existing index, or build only the missing corpus/index parts.

    ``force`` rebuilds the Chroma index even when the current collection opens
    successfully. It is used only by the explicit repair action in the UI.
    """
    if not force:
        started = time.perf_counter()
        ready = collection_ready(config)
        safe_log(
            logger,
            "kb_stage",
            stage="validate",
            status="reused" if ready else "missing",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        if ready:
            _emit(progress, "validate", 1.0, "Knowledge base ready.")
            return
    _build_missing_corpus(config, model, progress, force_index=force)
