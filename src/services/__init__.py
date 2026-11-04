"""Shared services used by both frontends.

The chat service is imported lazily so importing a lightweight submodule (for
example the knowledge-base readiness check) does not pull in the RAG pipeline.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ChatResult", "ChatUnavailableError", "process_question"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.services import chat

        return getattr(chat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
