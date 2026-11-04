"""Retrieve grounded chunks from the Chroma collection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from src.config import Config
from src.ingest.embed import load_model
from src.store.chroma import get_collection

PREFERRED_DOC_TYPES = {"kim", "sid", "factsheet", "scheme_page"}
LOWER_PRIORITY_DOC_TYPES = {"catalogue", "faq"}
SOURCE_PREFERENCE_WINDOW = 0.05
NEAR_DUPLICATE_RATIO = 0.95


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    debug: dict

    def to_dict(self) -> dict:
        return {"chunks": [chunk.to_dict() for chunk in self.chunks], "debug": self.debug}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%₹.]+", " ", text.lower())).strip()


def _is_near_duplicate(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= NEAR_DUPLICATE_RATIO


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    kept: list[RetrievedChunk] = []
    seen_hashes: set[str] = set()
    seen_text: list[str] = []
    for chunk in chunks:
        text_hash = chunk.metadata.get("text_hash", "")
        if text_hash and text_hash in seen_hashes:
            continue
        normalized = _normalize_text(chunk.text)
        if any(_is_near_duplicate(normalized, other) for other in seen_text):
            continue
        if text_hash:
            seen_hashes.add(text_hash)
        seen_text.append(normalized)
        kept.append(chunk)
    return kept


def _prefer_official_sources(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    ordered: list[RetrievedChunk] = []
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "")
        preferred = doc_type in PREFERRED_DOC_TYPES
        inserted = False
        if preferred:
            for position, existing in enumerate(ordered):
                existing_type = existing.metadata.get("doc_type", "")
                if (
                    existing_type not in PREFERRED_DOC_TYPES
                    and existing.score - chunk.score <= SOURCE_PREFERENCE_WINDOW
                ):
                    ordered.insert(position, chunk)
                    inserted = True
                    break
        if not inserted:
            ordered.append(chunk)
    return ordered


def _request_id(question: str, scheme: str | None) -> str:
    payload = f"{question.strip().lower()}|{scheme or ''}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def retrieve(
    config: Config,
    question: str,
    scheme: str | None = None,
    model=None,
    persist_debug: bool = False,
) -> RetrievalResult:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty")

    collection = get_collection(config)
    if model is None:
        model = load_model(config)

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    query = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=config.retrieval.top_k,
        where={"scheme": scheme} if scheme else None,
        include=["documents", "metadatas", "distances"],
    )

    documents = (query.get("documents") or [[]])[0]
    metadatas = (query.get("metadatas") or [[]])[0]
    distances = (query.get("distances") or [[]])[0]

    candidates: list[RetrievedChunk] = []
    below_threshold = 0
    for chunk_id, document, metadata, distance in zip(
        query["ids"][0], documents, metadatas, distances
    ):
        score = 1.0 - float(distance)
        if score < config.retrieval.score_threshold:
            below_threshold += 1
            continue
        candidates.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                score=score,
                text=document or "",
                metadata=dict(metadata or {}),
            )
        )

    candidates.sort(key=lambda chunk: chunk.score, reverse=True)
    before_dedupe = len(candidates)
    deduped = _dedupe(candidates)
    preferred = _prefer_official_sources(deduped)
    selected = preferred[: config.retrieval.max_context_chunks]

    debug = {
        "request_id": _request_id(question, scheme),
        "model": config.embedding.model,
        "top_k": config.retrieval.top_k,
        "score_threshold": config.retrieval.score_threshold,
        "scheme": scheme,
        "candidates": [
            {
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6),
                "source_id": chunk.metadata.get("source_id"),
                "url": chunk.url,
                "publisher": chunk.metadata.get("publisher"),
                "doc_type": chunk.metadata.get("doc_type"),
                "scheme": chunk.metadata.get("scheme"),
                "topic": chunk.metadata.get("topic"),
                "section": chunk.section,
            }
            for chunk in preferred
        ],
        "selected": [
            {
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6),
                "url": chunk.url,
                "section": chunk.section,
            }
            for chunk in selected
        ],
        "dropped": {
            "below_threshold": below_threshold,
            "duplicate": before_dedupe - len(deduped),
            "over_context_limit": max(0, len(preferred) - len(selected)),
        },
        "counts": {"candidates": len(preferred), "selected": len(selected)},
    }

    if persist_debug:
        out_dir = config.app.artifact_dir / "retrieval"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{debug['request_id']}.json"
        path.write_text(json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8")

    return RetrievalResult(chunks=selected, debug=debug)
