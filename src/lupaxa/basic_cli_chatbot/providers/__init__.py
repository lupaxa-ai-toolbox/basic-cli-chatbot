"""Provider registry. Not part of the public package API."""

from __future__ import annotations

from .base import Provider
from .errors import ChatError

KNOWN = ("openai", "gemini", "grok", "ollama")


def build(name: str, *, ollama_host: str, timeout: float) -> Provider:
    """Return the provider for a CLI name. This is the only name-to-class map."""
    key = name.strip().casefold()
    if key == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(timeout)
    if key == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(timeout)
    if key == "grok":
        from .grok import GrokProvider

        return GrokProvider(timeout)
    if key == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(ollama_host, timeout)
    joined = ", ".join(KNOWN)
    raise ChatError(f"unknown provider '{key}' ({joined})")


__all__ = ["KNOWN", "Provider", "build"]
