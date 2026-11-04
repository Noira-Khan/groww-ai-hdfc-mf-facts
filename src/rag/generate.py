"""Provider-agnostic LLM generation for grounded answers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from src.config import PROVIDER_KEY_ENV, Config
from src.rag.prompts import build_messages, build_repair_messages
from src.rag.retrieve import RetrievedChunk

OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GenerationError(RuntimeError):
    """Raised when the provider is misconfigured or returns unusable output."""


@dataclass
class GenerationOutput:
    payload: dict[str, Any]
    raw: str
    provider: str
    model: str


def _require_model(config: Config) -> str:
    model = (config.llm.model or "").strip()
    if not model:
        raise GenerationError(
            "LLM_MODEL_ID is empty. Set it in .env before calling the model."
        )
    return model


def _require_api_key(config: Config) -> str:
    key_env = PROVIDER_KEY_ENV.get(config.llm.provider)
    if not key_env:
        raise GenerationError(f"Unknown LLM provider: {config.llm.provider}")
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise GenerationError(f"{key_env} is not set.")
    return api_key


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    import httpx

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise GenerationError(
            f"Provider request failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise GenerationError("Provider request failed or timed out.") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise GenerationError("Provider returned non-JSON response.") from exc
    if not isinstance(data, dict):
        raise GenerationError("Provider returned an unexpected response.")
    return data


def _split_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    system_parts = [
        message["content"] for message in messages if message.get("role") == "system"
    ]
    user_parts = [
        message["content"] for message in messages if message.get("role") != "system"
    ]
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def _complete_openai_compatible(
    config: Config,
    messages: list[dict[str, str]],
    api_key: str,
    post_json: Callable[..., dict[str, Any]],
) -> str:
    url = f"{OPENAI_COMPATIBLE_BASE_URLS[config.llm.provider]}/chat/completions"
    payload = {
        "model": config.llm.model,
        "messages": messages,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
    }
    if config.llm.provider == "groq" and "gpt-oss" in config.llm.model.lower():
        payload["reasoning_effort"] = "low"
    data = post_json(
        url,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload,
        config.llm.timeout_s,
    )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GenerationError("Provider response did not contain message content.") from exc
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("Provider returned an empty message.")
    return content


def _complete_anthropic(
    config: Config,
    messages: list[dict[str, str]],
    api_key: str,
    post_json: Callable[..., dict[str, Any]],
) -> str:
    system, user = _split_messages(messages)
    data = post_json(
        ANTHROPIC_URL,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        {
            "model": config.llm.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
        config.llm.timeout_s,
    )
    try:
        parts = data["content"]
        content = "".join(
            part.get("text", "") for part in parts if part.get("type") == "text"
        )
    except (KeyError, TypeError) as exc:
        raise GenerationError("Provider response did not contain message content.") from exc
    if not content.strip():
        raise GenerationError("Provider returned an empty message.")
    return content


def _complete_gemini(
    config: Config,
    messages: list[dict[str, str]],
    api_key: str,
    post_json: Callable[..., dict[str, Any]],
) -> str:
    system, user = _split_messages(messages)
    model = config.llm.model.removeprefix("models/")
    data = post_json(
        GEMINI_URL.format(model=model),
        {"x-goog-api-key": api_key, "Content-Type": "application/json"},
        {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": config.llm.temperature,
                "maxOutputTokens": config.llm.max_tokens,
            },
        },
        config.llm.timeout_s,
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
        content = "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise GenerationError("Provider response did not contain message content.") from exc
    if not content.strip():
        raise GenerationError("Provider returned an empty message.")
    return content


def complete(
    config: Config,
    messages: list[dict[str, str]],
    post_json: Callable[..., dict[str, Any]] = _post_json,
) -> str:
    """Send messages to the configured provider and return text content."""
    _require_model(config)
    api_key = _require_api_key(config)

    if config.llm.provider in OPENAI_COMPATIBLE_BASE_URLS:
        return _complete_openai_compatible(config, messages, api_key, post_json)
    if config.llm.provider == "anthropic":
        return _complete_anthropic(config, messages, api_key, post_json)
    if config.llm.provider == "gemini":
        return _complete_gemini(config, messages, api_key, post_json)
    raise GenerationError(f"Unknown LLM provider: {config.llm.provider}")


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise GenerationError("Provider returned invalid JSON (no object found).")
    return text[start : end + 1]


def parse_candidate(raw: str) -> dict[str, Any]:
    """Parse the provider's strict-JSON answer, tolerating code fences."""
    try:
        payload = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise GenerationError("Provider returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise GenerationError("Provider JSON must be an object.")

    if "not_in_sources" in payload:
        payload["not_in_sources"] = bool(payload["not_in_sources"])
        return payload

    for key in ("answer", "source_url", "proof"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise GenerationError(f"Provider JSON is missing a non-empty '{key}'.")
    return payload


def generate_candidate(
    config: Config,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    repair_candidate: str | None = None,
    repair_issues: list[str] | None = None,
    complete_fn: Callable[[Config, list[dict[str, str]]], str] | None = None,
    post_json: Callable[..., dict[str, Any]] = _post_json,
) -> GenerationOutput:
    _require_model(config)
    if not chunks:
        raise GenerationError("Cannot generate an answer without retrieved context.")

    if repair_candidate is not None:
        messages = build_repair_messages(
            question, chunks, repair_candidate, repair_issues or []
        )
    else:
        messages = build_messages(question, chunks)

    if complete_fn is not None:
        raw = complete_fn(config, messages)
    else:
        raw = complete(config, messages, post_json=post_json)

    return GenerationOutput(
        payload=parse_candidate(raw),
        raw=raw,
        provider=config.llm.provider,
        model=config.llm.model,
    )
