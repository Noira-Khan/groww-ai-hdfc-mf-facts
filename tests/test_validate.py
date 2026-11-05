from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.rag.retrieve import RetrievedChunk
from src.rag.validate import (
    count_sentences,
    extract_urls,
    is_allowlisted_url,
    validate_candidate,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
URL = "https://files.hdfcfund.com/KIM/flexi-cap.pdf"
TEXT = "Exit load is 1% for redemptions within one year."


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.project_root = tmp_path
    cfg.app.artifact_dir = tmp_path / "artifacts"
    return cfg


def _chunk(
    chunk_id: str = "c1",
    text: str = TEXT,
    url: str = URL,
    publisher: str = "HDFC AMC",
    as_on_date: str = "2026-01-02",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.9,
        text=text,
        metadata={
            "url": url,
            "publisher": publisher,
            "doc_type": "kim",
            "scheme": "flexi_cap",
            "topic": "exit_load",
            "section": "Exit load",
            "as_on_date": as_on_date,
        },
    )


def _candidate(**overrides):
    value = {
        "answer": f"The exit load of HDFC Flexi Cap Fund is 1%. Source: {URL}",
        "source_url": URL,
        "proof": "Exit load is 1%",
    }
    value.update(overrides)
    return value


def test_valid_candidate(config):
    result = validate_candidate(
        config, _candidate(), [_chunk()], "What is the exit load?"
    )

    assert result.ok is True
    assert result.issues == []
    assert result.source_name == "HDFC AMC"
    assert result.as_on_date == "2026-01-02"


def test_rejects_too_many_sentences(config):
    answer = (
        f"One. Two. Three. Four. Source: {URL}"
    )
    result = validate_candidate(
        config, _candidate(answer=answer), [_chunk()], "exit load"
    )

    assert not result.ok
    assert "too_many_sentences" in {issue.code for issue in result.issues}


def test_rejects_wrong_url_count(config):
    no_url = validate_candidate(
        config,
        _candidate(answer="The exit load is 1%.", source_url=""),
        [_chunk()],
        "exit load",
    )
    assert "url_count" in {issue.code for issue in no_url.issues}
    assert "url_missing" in {issue.code for issue in no_url.issues}

    two_urls = validate_candidate(
        config,
        _candidate(answer=f"See {URL} and https://www.amfiindia.com/x"),
        [_chunk()],
        "exit load",
    )
    assert "url_count" in {issue.code for issue in two_urls.issues}


def test_url_may_be_carried_only_in_source_url(config):
    result = validate_candidate(
        config,
        _candidate(answer="The exit load is 1%."),
        [_chunk()],
        "What is the exit load?",
    )
    assert result.ok is True


def test_rejects_non_allowlisted_or_unretrieved_url(config):
    bad_host = validate_candidate(
        config,
        _candidate(
            answer="The exit load is 1%. Source: https://example.com/x",
            source_url="https://example.com/x",
        ),
        [_chunk()],
        "exit load",
    )
    assert "url_not_allowlisted" in {issue.code for issue in bad_host.issues}

    other_official = "https://www.hdfcfund.com/other.pdf"
    not_retrieved = validate_candidate(
        config,
        _candidate(
            answer=f"The exit load is 1%. Source: {other_official}",
            source_url=other_official,
        ),
        [_chunk()],
        "exit load",
    )
    assert "url_not_retrieved" in {issue.code for issue in not_retrieved.issues}


def test_rejects_non_allowlisted_publisher(config):
    result = validate_candidate(
        config, _candidate(), [_chunk(publisher="Moneycontrol")], "exit load"
    )
    assert "publisher_not_allowlisted" in {issue.code for issue in result.issues}


def test_rejects_advice_and_performance_phrases(config):
    advice = validate_candidate(
        config,
        _candidate(answer=f"You should buy this fund. Source: {URL}"),
        [_chunk()],
        "exit load",
    )
    assert "advice_phrase" in {issue.code for issue in advice.issues}

    performance = validate_candidate(
        config,
        _candidate(answer=f"The fund's 3-year returns are strong. Source: {URL}"),
        [_chunk()],
        "exit load",
    )
    assert "performance_phrase" in {issue.code for issue in performance.issues}


def test_rejects_ungrounded_number(config):
    result = validate_candidate(
        config,
        _candidate(answer=f"The exit load is 2%. Source: {URL}"),
        [_chunk()],
        "What is the exit load?",
    )
    assert "ungrounded_number" in {issue.code for issue in result.issues}


def test_accepts_equivalent_number_formatting(config):
    result = validate_candidate(
        config,
        _candidate(
            answer=f"The exit load is 1.00%. Source: {URL}",
            proof="Exit load is 1%",
        ),
        [_chunk()],
        "What is the exit load?",
    )
    assert result.ok is True


def test_number_can_come_from_question(config):
    result = validate_candidate(
        config,
        _candidate(
            answer=f"The lock-in is 3 years. Source: {URL}",
            proof="Exit load is 1%",
        ),
        [_chunk()],
        "Is the lock-in 3 years?",
    )
    assert "ungrounded_number" not in {issue.code for issue in result.issues}


def test_rejects_paraphrased_proof(config):
    result = validate_candidate(
        config,
        _candidate(proof="Exit load is one percent"),
        [_chunk()],
        "exit load",
    )
    assert "proof_not_verbatim" in {issue.code for issue in result.issues}


def test_proof_must_come_from_cited_url(config):
    cited = _chunk("c1", text="A different passage.")
    other = _chunk(
        "c2",
        text=TEXT,
        url="https://www.hdfcfund.com/other.pdf",
    )
    result = validate_candidate(
        config,
        _candidate(),
        [cited, other],
        "exit load",
    )
    assert "proof_not_verbatim" in {issue.code for issue in result.issues}


def test_proof_can_come_from_any_chunk_sharing_cited_url(config):
    first = _chunk("c1", text="Intro text only.")
    second = _chunk("c2", text=TEXT)
    result = validate_candidate(
        config,
        _candidate(),
        [first, second],
        "exit load",
    )
    assert result.ok is True
    assert result.issues == []


def test_rejects_pii_echo(config):
    result = validate_candidate(
        config,
        _candidate(answer=f"My PAN is ABCDE1234F. Source: {URL}"),
        [_chunk()],
        "My PAN is ABCDE1234F",
    )
    assert "pii_echo" in {issue.code for issue in result.issues}


def test_not_in_sources_candidate(config):
    result = validate_candidate(config, {"not_in_sources": True}, [_chunk()], "x")
    assert result.ok is False
    assert result.not_in_sources is True


def test_sentence_count_handles_abbreviations_and_urls():
    assert count_sentences("The minimum SIP is Rs. 500 per month.") == 1
    assert count_sentences(f"One. Two. See {URL}.") == 3
    assert count_sentences("") == 0


def test_extract_urls_trims_sentence_punctuation():
    assert extract_urls(f"See {URL}.") == [URL]


def test_allowlisted_hosts():
    assert is_allowlisted_url("https://www.hdfcfund.com/x")
    assert is_allowlisted_url("https://files.hdfcfund.com/x")
    assert is_allowlisted_url("https://investor.sebi.gov.in/x")
    assert is_allowlisted_url("https://www.amfiindia.com/x")
    assert is_allowlisted_url("https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth")
    assert not is_allowlisted_url("javascript:alert(1)")
