"""User-facing chat failures. Not part of the public package API."""

from __future__ import annotations


class ChatError(Exception):
    """Failure whose message is printed after ``Error: ``."""


def one_line(exc: BaseException) -> str:
    """Return the first line of an exception, capped at 200 characters."""
    raw = str(exc).strip()
    if not raw:
        return "the provider returned an invalid response"
    return raw.splitlines()[0][:200]


def looks_like_context_limit(text: str) -> bool:
    """Return whether a provider message is a context-window rejection."""
    lowered = text.lower()
    return "context" in lowered and any(word in lowered for word in ("length", "window", "token"))
