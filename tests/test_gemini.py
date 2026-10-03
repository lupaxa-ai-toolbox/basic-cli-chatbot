"""Tests for the Gemini provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from google.genai import errors, types

from lupaxa.basic_cli_chatbot.chat import SYSTEM_PROMPT
from lupaxa.basic_cli_chatbot.providers.errors import ChatError
from lupaxa.basic_cli_chatbot.providers.gemini import GeminiProvider

_ENV_VARS = (
    "BASIC_CLI_CHATBOT_PROVIDER",
    "BASIC_CLI_CHATBOT_MODEL",
    "BASIC_CLI_CHATBOT_TIMEOUT",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


class _Models:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self._text = "four"
        self._error: BaseException | None = None
        self._chunks = ["fo", "ur"]

    def generate_content(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _Response(self._text)

    def generate_content_stream(self, **kwargs: object):
        self.stream_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield _Response(chunk)

    def list(self):
        yield type("Model", (), {"name": "models/gemini-a"})()
        yield type("Model", (), {"name": "gemini-b"})()


class _Client:
    def __init__(self) -> None:
        self.models = _Models()


def _provider(monkeypatch: pytest.MonkeyPatch, client: _Client) -> GeminiProvider:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.gemini.genai.Client",
        lambda **_kwargs: client,
    )
    return GeminiProvider(60)


def test_chat_uses_system_instruction_and_model_role(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    provider = _provider(monkeypatch, client)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "next"},
    ]
    assert provider.chat(messages, "some-model", SYSTEM_PROMPT) == "four"
    call = client.models.calls[0]
    assert call["model"] == "some-model"
    assert call["config"].system_instruction == SYSTEM_PROMPT
    assert call["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
        {"role": "user", "parts": [{"text": "next"}]},
    ]


def test_stream_yields_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, _Client())
    assert list(provider.stream([], "some-model", SYSTEM_PROMPT)) == ["fo", "ur"]


def test_list_models_strips_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, _Client())
    assert provider.list_models() == ["gemini-a", "gemini-b"]


def test_timeout_is_milliseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_client(**kwargs: object) -> _Client:
        seen.update(kwargs)
        return _Client()

    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.gemini.genai.Client",
        fake_client,
    )
    assert GeminiProvider(30.5).list_models() == ["gemini-a", "gemini-b"]
    options = seen["http_options"]
    assert isinstance(options, types.HttpOptions)
    assert options.timeout == 30500


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ChatError, match="GEMINI_API_KEY is not configured"):
        GeminiProvider(60).chat([], "m", SYSTEM_PROMPT)


@pytest.mark.parametrize(
    ("code", "message", "match"),
    [
        (401, "denied", "Gemini authentication failed"),
        (403, "denied", "Gemini authentication failed"),
        (429, "slow down", "Gemini rate limit reached"),
        (404, "missing", "model 'some-model' is not available"),
        (500, "down", "Gemini is unavailable"),
        (400, "maximum context length exceeded", "the conversation is too long for this model"),
    ],
)
def test_catalogue_errors(
    monkeypatch: pytest.MonkeyPatch, code: int, message: str, match: str
) -> None:
    client = _Client()
    client.models._error = errors.APIError(code, {"error": {"message": message, "code": code}})
    provider = _provider(monkeypatch, client)
    with pytest.raises(ChatError, match=match):
        provider.chat([], "some-model", SYSTEM_PROMPT)


def test_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    client.models._error = httpx.ConnectError("refused")
    provider = _provider(monkeypatch, client)
    with pytest.raises(ChatError, match="network error talking to Gemini"):
        provider.chat([], "some-model", SYSTEM_PROMPT)
