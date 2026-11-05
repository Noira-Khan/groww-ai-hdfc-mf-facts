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
from src.rag.topics import detect_topic, text_matches_topic
from src.safety.intent import SUPPORTED_SCHEMES
from src.store.chroma import get_collection

PREFERRED_DOC_TYPES = {"kim", "sid", "factsheet", "scheme_page"}
LOWER_PRIORITY_DOC_TYPES = {"catalogue", "faq", "education"}
AMC_WIDE_SCHEMES = frozenset({"", "amc_wide", "all", "general", "none"})
TOPIC_PREFERRED_DOC_TYPES: dict[str, set[str]] = {
    "statements": {"statement_guide", "faq"},
    "capital_gains": {"statement_guide", "faq"},
    "definitions": {"faq", "education", "scheme_page"},
}
SOURCE_PREFERENCE_WINDOW = 0.05
NEAR_DUPLICATE_RATIO = 0.95
LEXICAL_BOOST_MAX = 0.05
LOWER_PRIORITY_PENALTY = 0.03
CROSS_SCHEME_PENALTY = 0.10
TOPIC_DOC_BOOST = 0.06
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "do",
        "for",
        "how",
        "i",
        "in",
        "is",
        "my",
        "of",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)
    raw_score: float | None = None

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    def to_dict(self) -> dict:
        payload = {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.raw_score is not None:
            payload["raw_score"] = round(self.raw_score, 6)
        return payload


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


def _preferred_doc_types(detected_topic: str | None) -> set[str]:
    preferred = set(PREFERRED_DOC_TYPES)
    if detected_topic:
        preferred |= TOPIC_PREFERRED_DOC_TYPES.get(detected_topic, set())
    return preferred


def _prefer_official_sources(
    chunks: list[RetrievedChunk],
    detected_topic: str | None = None,
) -> list[RetrievedChunk]:
    preferred_types = _preferred_doc_types(detected_topic)
    ordered: list[RetrievedChunk] = []
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "")
        preferred = doc_type in preferred_types
        inserted = False
        if preferred:
            for position, existing in enumerate(ordered):
                existing_type = existing.metadata.get("doc_type", "")
                if (
                    existing_type not in preferred_types
                    and existing.score - chunk.score <= SOURCE_PREFERENCE_WINDOW
                ):
                    ordered.insert(position, chunk)
                    inserted = True
                    break
        if not inserted:
            ordered.append(chunk)
    return ordered


def _cross_scheme_penalty(chunk: RetrievedChunk, query_scheme: str | None) -> float:
    """When the question has no scheme, demote scheme-specific KIM/page chunks."""
    if query_scheme:
        return 0.0
    meta_scheme = str(chunk.metadata.get("scheme") or "").strip().lower()
    if meta_scheme in AMC_WIDE_SCHEMES:
        return 0.0
    if meta_scheme in SUPPORTED_SCHEMES or meta_scheme == "small_cap":
        return CROSS_SCHEME_PENALTY
    return 0.0


def _topic_doc_boost(chunk: RetrievedChunk, detected_topic: str | None) -> float:
    if not detected_topic:
        return 0.0
    preferred = TOPIC_PREFERRED_DOC_TYPES.get(detected_topic)
    if not preferred:
        return 0.0
    if chunk.metadata.get("doc_type") in preferred:
        return TOPIC_DOC_BOOST
    return 0.0


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _lexical_boost(question: str, chunk: RetrievedChunk) -> float:
    query_tokens = _content_tokens(question)
    if not query_tokens:
        return 0.0
    chunk_tokens = _content_tokens(f"{chunk.section} {chunk.text}")
    if not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
    return LEXICAL_BOOST_MAX * overlap


def _matches_topic(chunk: RetrievedChunk, topic: str | None) -> bool:
    if not topic:
        return False
    meta_topic = chunk.metadata.get("topic")
    if meta_topic == topic:
        return True
    # Chunk body is the strongest fallback when metadata is missing or generic.
    if text_matches_topic(topic, chunk.text):
        return True
    if meta_topic in (None, "", "general") and text_matches_topic(
        topic, chunk.section
    ):
        return True
    return False


def _select_context(
    chunks: list[RetrievedChunk],
    limit: int,
    detected_topic: str | None = None,
) -> list[RetrievedChunk]:
    """Pick top context with source diversity, preferring topic-matched evidence."""
    if limit <= 0 or not chunks:
        return []

    selected: list[RetrievedChunk] = []
    seen_sources: set[str] = set()
    preferred_types = _preferred_doc_types(detected_topic)

    def _source_key(chunk: RetrievedChunk) -> str:
        return str(chunk.metadata.get("source_id") or chunk.chunk_id)

    def _take(pool: list[RetrievedChunk], *, unique_source: bool) -> None:
        for chunk in pool:
            if len(selected) >= limit:
                return
            if chunk in selected:
                continue
            key = _source_key(chunk)
            if unique_source and key in seen_sources:
                continue
            selected.append(chunk)
            seen_sources.add(key)

    topic_hits = [chunk for chunk in chunks if _matches_topic(chunk, detected_topic)]
    preferred_hits = [
        chunk
        for chunk in topic_hits
        if chunk.metadata.get("doc_type") in preferred_types
    ]
    other_hits = [chunk for chunk in topic_hits if chunk not in preferred_hits]
    non_hits = [chunk for chunk in chunks if chunk not in topic_hits]

    # Exhaust preferred topic docs before weaker topic/non-topic fillers.
    _take(preferred_hits, unique_source=True)
    _take(preferred_hits, unique_source=False)
    _take(other_hits, unique_source=True)
    _take(other_hits, unique_source=False)
    _take(non_hits, unique_source=True)
    _take(chunks, unique_source=False)
    return selected


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
        n_results=max(config.retrieval.top_k, config.retrieval.candidate_pool),
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
                raw_score=score,
            )
        )

    detected_topic = detect_topic(question)
    for chunk in candidates:
        doc_type = chunk.metadata.get("doc_type", "")
        if doc_type in LOWER_PRIORITY_DOC_TYPES:
            chunk.score -= LOWER_PRIORITY_PENALTY
        chunk.score -= _cross_scheme_penalty(chunk, scheme)
        chunk.score += _topic_doc_boost(chunk, detected_topic)
        chunk.score += _lexical_boost(question, chunk)
        if detected_topic and config.retrieval.topic_boost > 0:
            if _matches_topic(chunk, detected_topic):
                chunk.score += config.retrieval.topic_boost

    candidates.sort(key=lambda chunk: chunk.score, reverse=True)
    before_dedupe = len(candidates)
    deduped = _dedupe(candidates)
    preferred = _prefer_official_sources(deduped, detected_topic=detected_topic)
    selected = _select_context(
        preferred,
        config.retrieval.max_context_chunks,
        detected_topic=detected_topic,
    )

    debug = {
        "request_id": _request_id(question, scheme),
        "model": config.embedding.model,
        "top_k": config.retrieval.top_k,
        "candidate_pool": max(
            config.retrieval.top_k, config.retrieval.candidate_pool
        ),
        "score_threshold": config.retrieval.score_threshold,
        "scheme": scheme,
        "detected_topic": detected_topic,
        "topic_boost": config.retrieval.topic_boost,
        "candidates": [
            {
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6),
                "raw_score": round(
                    chunk.raw_score if chunk.raw_score is not None else chunk.score,
                    6,
                ),
                "source_id": chunk.metadata.get("source_id"),
                "url": chunk.url,
                "publisher": chunk.metadata.get("publisher"),
                "doc_type": chunk.metadata.get("doc_type"),
                "scheme": chunk.metadata.get("scheme"),
                "topic": chunk.metadata.get("topic"),
                "topic_match": (
                    _matches_topic(chunk, detected_topic) if detected_topic else None
                ),
                "section": chunk.section,
            }
            for chunk in preferred
        ],
        "selected": [
            {
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6),
                "source_id": chunk.metadata.get("source_id"),
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
