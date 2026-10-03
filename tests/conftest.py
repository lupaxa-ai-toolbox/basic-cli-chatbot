"""Suite-wide isolation. Tests must not read the developer's config file."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _missing_default_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-config.yaml"
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.cli.default_config_path",
        lambda: missing,
    )
