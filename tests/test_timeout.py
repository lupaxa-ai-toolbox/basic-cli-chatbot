"""Tests for reply-timeout conversion. No SDK imports."""

import pytest

from lupaxa.basic_cli_chatbot.providers.catalog import CATALOG, gemini_timeout_ms


def test_catalog_defaults() -> None:
    assert CATALOG["openai"].timeout_seconds == 60
    assert CATALOG["gemini"].timeout_seconds == 60
    assert CATALOG["ollama"].timeout_seconds == 300
    assert CATALOG["grok"].mode == "Remote"
    assert CATALOG["grok"].api_key_env == "XAI_API_KEY"
    assert CATALOG["grok"].timeout_seconds == 60


def test_gemini_timeout_ms() -> None:
    assert gemini_timeout_ms(30) == 30000
    assert gemini_timeout_ms(30.5) == 30500
    assert gemini_timeout_ms(0.0006) == 1
    assert gemini_timeout_ms(0.0015) == 2


def test_gemini_timeout_ms_rejects_a_rounded_zero() -> None:
    message = "timeout must be at least 1 millisecond for Gemini"
    with pytest.raises(ValueError, match=f"^{message}$"):
        gemini_timeout_ms(0.0004)
    with pytest.raises(ValueError, match=f"^{message}$"):
        gemini_timeout_ms(0.0005)
