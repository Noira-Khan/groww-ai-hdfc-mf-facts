"""Curated questions that are verified against the indexed corpus.

The first eight entries are the factual evaluation questions (F01–F08).
They have strong retrieval coverage and are the prompts shown in Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suggestion:
    question: str
    topic: str | None
    scheme: str | None = None
    min_score: float = 0.55


# Factual evaluation questions (eval/gold_questions.csv F01–F08), retrieval-verified.
CURATED_SUGGESTIONS: tuple[Suggestion, ...] = (
    Suggestion(
        "What is the exit load of HDFC Flexi Cap Fund?",
        "exit_load",
        "flexi_cap",
    ),
    Suggestion(
        "What is the lock-in period of HDFC ELSS Tax Saver?",
        "elss_lockin",
        "elss",
    ),
    Suggestion(
        "What is the minimum SIP for HDFC Large Cap Fund?",
        "min_sip",
        "large_cap",
    ),
    Suggestion(
        "What is the Total Expense Ratio of HDFC Mid Cap Fund?",
        "expense_ratio",
        "mid_cap",
    ),
    Suggestion(
        "What is the riskometer of HDFC Large Cap Fund?",
        "riskometer",
        "large_cap",
    ),
    Suggestion(
        "What is the benchmark of HDFC Mid Cap Fund?",
        "benchmark",
        "mid_cap",
    ),
    Suggestion(
        "How do I request an account statement?",
        "statements",
    ),
    Suggestion(
        "What is the riskometer of HDFC ELSS Tax Saver?",
        "riskometer",
        "elss",
    ),
)

SAFE_QUESTIONS: tuple[str, ...] = tuple(
    suggestion.question for suggestion in CURATED_SUGGESTIONS[:3]
)

DEFAULT_SUGGESTIONS: tuple[str, ...] = tuple(
    suggestion.question for suggestion in CURATED_SUGGESTIONS
)
