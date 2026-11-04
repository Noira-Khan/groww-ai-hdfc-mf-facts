"""Shared exception types with safe user-facing messages."""

from __future__ import annotations


class AppError(Exception):
    """Base application error.

    `user_message` is safe to show in the UI.
    `detail` is for internal logs only and must never include secrets or PII.
    """

    user_message = "Something went wrong. Please try again."

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail or self.user_message)
        self.detail = detail


class ConfigError(AppError):
    user_message = "The assistant is not configured correctly."


class IngestionError(AppError):
    user_message = "A source document could not be processed."


class RetrievalError(AppError):
    user_message = "The document index is unavailable."


class ValidationError(AppError):
    user_message = "I could not find a supported answer in the official sources."


class ProviderError(AppError):
    user_message = "The answer service is temporarily unavailable."
