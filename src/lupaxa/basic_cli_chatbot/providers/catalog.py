"""Provider facts that do not load an SDK. Not part of the public package API."""

from __future__ import annotations

from typing import NamedTuple


class ProviderFacts(NamedTuple):
    """Mode, optional API-key variable, and built-in reply timeout in seconds."""

    mode: str
    api_key_env: str | None
    timeout_seconds: float


CATALOG: dict[str, ProviderFacts] = {
    "openai": ProviderFacts(
        mode="Remote",
        api_key_env="OPENAI_API_KEY",
        timeout_seconds=60,
    ),
    "gemini": ProviderFacts(
        mode="Remote",
        api_key_env="GEMINI_API_KEY",
        timeout_seconds=60,
    ),
    "grok": ProviderFacts(
        mode="Remote",
        api_key_env="XAI_API_KEY",
        timeout_seconds=60,
    ),
    "ollama": ProviderFacts(mode="Local", api_key_env=None, timeout_seconds=300),
}


def gemini_timeout_ms(seconds: float) -> int:
    """Convert seconds to Gemini's millisecond timeout."""
    milliseconds = round(seconds * 1000)
    if milliseconds < 1:
        raise ValueError("timeout must be at least 1 millisecond for Gemini")
    return milliseconds
