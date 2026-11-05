"""Curated questions that are verified against the indexed corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suggestion:
    question: str
    topic: str | None
    scheme: str | None = None
    min_score: float = 0.55


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
        "What is the riskometer of HDFC Mid Cap Fund?",
        "riskometer",
        "mid_cap",
    ),
    Suggestion(
        "What is the benchmark of HDFC Large Cap Fund?",
        "benchmark",
        "large_cap",
    ),
    Suggestion(
        "What is the minimum SIP for HDFC Large Cap Fund?",
        "min_sip",
        "large_cap",
    ),
    Suggestion(
        "How do I request an account statement?",
        "statements",
    ),
    Suggestion(
        "What is NAV in a mutual fund?",
        "definitions",
    ),
    Suggestion(
        "What is the Total Expense Ratio of HDFC Mid Cap Fund?",
        "expense_ratio",
        "mid_cap",
    ),
    Suggestion(
        "What is the exit load of HDFC Large Cap Fund?",
        "exit_load",
        "large_cap",
    ),
    Suggestion(
        "What is the riskometer of HDFC Flexi Cap Fund?",
        "riskometer",
        "flexi_cap",
    ),
    Suggestion(
        "What is the benchmark of HDFC Mid Cap Fund?",
        "benchmark",
        "mid_cap",
    ),
    Suggestion(
        "What is the minimum SIP for HDFC Flexi Cap Fund?",
        "min_sip",
        "flexi_cap",
    ),
    Suggestion(
        "What is the Total Expense Ratio of HDFC Flexi Cap Fund?",
        "expense_ratio",
        "flexi_cap",
    ),
    Suggestion(
        "What is the exit load of HDFC Mid Cap Fund?",
        "exit_load",
        "mid_cap",
    ),
)

SAFE_QUESTIONS: tuple[str, ...] = tuple(
    suggestion.question for suggestion in CURATED_SUGGESTIONS[:3]
)

DEFAULT_SUGGESTIONS: tuple[str, ...] = tuple(
    suggestion.question for suggestion in CURATED_SUGGESTIONS[:6]
)
