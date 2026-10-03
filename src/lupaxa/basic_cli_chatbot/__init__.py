"""Command-line AI chatbot. Public names are version symbols only."""

from __future__ import annotations

from .version import __version__, get_version

__all__ = [
    "__version__",
    "get_version",
]
