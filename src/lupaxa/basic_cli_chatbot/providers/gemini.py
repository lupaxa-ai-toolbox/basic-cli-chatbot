"""Gemini provider. Not part of the public package API."""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
from google import genai
from google.genai import errors, types

from .catalog import CATALOG, gemini_timeout_ms
from .errors import (
    ChatError,
    looks_like_context_limit,
    one_line,
)


class GeminiProvider:
    """Chat through the Google GenAI SDK."""

    mode = CATALOG["gemini"].mode
    supports_stream = True

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> str:
        client = _client(self.timeout)
        try:
            response = client.models.generate_content(
                model=model,
                contents=_contents(messages),
                config=_config(system_prompt),
            )
        except Exception as exc:
            raise _guard(exc, model) from None
        raw = response.text
        if raw is None or not raw.strip():
            raise ChatError("the provider returned an invalid response")
        return raw

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> Iterator[str]:
        client = _client(self.timeout)
        try:
            chunks = client.models.generate_content_stream(
                model=model,
                contents=_contents(messages),
                config=_config(system_prompt),
            )
            for chunk in chunks:
                text = chunk.text or ""
                if text:
                    yield text
        except Exception as exc:
            raise _guard(exc, model) from None

    def list_models(self) -> list[str]:
        client = _client(self.timeout)
        try:
            names: list[str] = []
            for item in client.models.list():
                name = item.name or ""
                if name.startswith("models/"):
                    name = name[len("models/") :]
                if name:
                    names.append(name)
            return names
        except Exception as exc:
            raise _guard(exc, "") from None


def _client(timeout: float) -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ChatError("GEMINI_API_KEY is not configured")
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=gemini_timeout_ms(timeout)),
    )


def _config(system_prompt: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(system_instruction=system_prompt)


def _contents(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    contents: list[dict[str, object]] = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return contents


def _guard(exc: Exception, model: str) -> ChatError:
    if isinstance(exc, errors.APIError):
        return _translate(exc, model)
    if isinstance(exc, httpx.TimeoutException):
        return ChatError("Gemini request timed out")
    if isinstance(exc, httpx.TransportError):
        return ChatError("network error talking to Gemini")
    return ChatError(one_line(exc))


def _translate(exc: errors.APIError, model: str) -> ChatError:
    code = exc.code
    detail = exc.message or str(exc)
    if code in {401, 403}:
        return ChatError("Gemini authentication failed")
    if code == 429:
        return ChatError("Gemini rate limit reached")
    if code == 404:
        return ChatError(f"model '{model}' is not available")
    if code is not None and code >= 500:
        return ChatError("Gemini is unavailable")
    if code == 400 and looks_like_context_limit(detail):
        return ChatError("the conversation is too long for this model")
    if "timed out" in detail.lower() or "timeout" in detail.lower():
        return ChatError("Gemini request timed out")
    return ChatError(one_line(exc))
