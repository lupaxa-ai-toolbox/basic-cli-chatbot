"""Provider interface. Not part of the public package API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class Provider(Protocol):
    """One chat backend selected for the process."""

    mode: str
    supports_stream: bool

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> str:
        """Return the full assistant reply."""
        ...

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> Iterator[str]:
        """Yield reply text deltas."""
        ...

    def list_models(self) -> list[str]:
        """Return model names accepted by ``--model``."""
        ...
