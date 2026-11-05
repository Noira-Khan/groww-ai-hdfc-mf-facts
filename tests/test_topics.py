from __future__ import annotations

from src.rag.topics import (
    SCHEME_SENSITIVE_TOPICS,
    detect_topic,
    scheme_topic_question,
)


def test_detects_fact_topics():
    assert detect_topic("What is the exit load?") == "exit_load"
    assert detect_topic("What is the exit-load?") == "exit_load"
    assert detect_topic("What is the lock-in period?") == "elss_lockin"
    assert detect_topic("What is the minimum SIP?") == "min_sip"
    assert detect_topic("What is the Total Expense Ratio?") == "expense_ratio"
    assert detect_topic("What is the riskometer?") == "riskometer"
    assert detect_topic("What is the benchmark?") == "benchmark"
    assert detect_topic("How do I request an account statement?") == "statements"
    assert detect_topic("How do I download my capital-gains statement?") == (
        "capital_gains"
    )
    assert detect_topic("What is NAV in a mutual fund?") == "definitions"
    assert detect_topic("Tell me about HDFC Mid Cap Fund.") is None


def test_text_matches_topic_reads_chunk_body():
    from src.rag.topics import text_matches_topic

    assert text_matches_topic(
        "exit_load", "Fees", "An Exit Load of 1.00% applies."
    )
    assert not text_matches_topic(
        "exit_load", "Fees", "Minimum SIP is 100."
    )


def test_builds_self_contained_scheme_followups():
    assert scheme_topic_question("exit_load", "mid_cap") == (
        "What is the exit load of HDFC Mid Cap Fund?"
    )
    assert scheme_topic_question("elss_lockin", "elss") == (
        "What is the lock-in period of HDFC ELSS Tax Saver?"
    )
    assert scheme_topic_question("definitions", "mid_cap") is None
    assert scheme_topic_question("exit_load", "small_cap") is None
    assert "exit_load" in SCHEME_SENSITIVE_TOPICS
