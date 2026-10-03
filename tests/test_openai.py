"""Tests for the OpenAI provider."""

from __future__ import annotations

from typing import Any, Literal, cast

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from lupaxa.basic_cli_chatbot.chat import SYSTEM_PROMPT
from lupaxa.basic_cli_chatbot.providers.errors import ChatError
from lupaxa.basic_cli_chatbot.providers.openai import OpenAIProvider

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
        self.output_text = text


class _Stream:
    def __init__(self, deltas: list[str | BaseException]) -> None:
        self._deltas = deltas

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_args: object) -> Literal[False]:
        return False

    def __iter__(self):
        for delta in self._deltas:
            if isinstance(delta, BaseException):
                raise delta
            event = type("Event", (), {"type": "response.output_text.delta", "delta": delta})()
            yield event


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []
        self._text = "four"
        self._error: BaseException | None = None
        self._deltas: list[str | BaseException] = ["fo", "ur"]

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _Response(self._text)

    def stream(self, **kwargs: object) -> _Stream:
        self.stream_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _Stream(self._deltas)


class _Models:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def list(self, **_kwargs: object):
        for model_id in self._ids:
            yield type("Model", (), {"id": model_id})()


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()
        self.models = _Models(["gpt-a", "gpt-b"])


def _provider(
    monkeypatch: pytest.MonkeyPatch,
    client: _Client,
    timeout: float = 60.0,
) -> OpenAIProvider:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.openai.OpenAI",
        lambda **_kwargs: client,
    )
    return OpenAIProvider(timeout)


def test_chat_sends_a_snapshot_and_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    provider = _provider(monkeypatch, client)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    assert provider.chat(messages, "some-model", SYSTEM_PROMPT) == "four"
    messages.append({"role": "assistant", "content": "mutated"})
    call = client.responses.calls[0]
    assert call["model"] == "some-model"
    assert call["instructions"] == SYSTEM_PROMPT
    assert call["timeout"] == 60.0
    assert call["input"] == [{"role": "user", "content": "What is 2+2?"}]


def test_stream_yields_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    provider = _provider(monkeypatch, client)
    assert list(provider.stream([], "some-model", SYSTEM_PROMPT)) == ["fo", "ur"]


def test_list_models_returns_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, _Client())
    assert provider.list_models() == ["gpt-a", "gpt-b"]


def test_timeout_is_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    seen: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> _Client:
        seen.update(kwargs)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.openai.OpenAI",
        fake_openai,
    )
    provider = OpenAIProvider(30.5)
    assert provider.chat([], "m", SYSTEM_PROMPT) == "four"
    assert seen["timeout"] == 30.5
    assert client.responses.calls[0]["timeout"] == 30.5


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ChatError, match="OPENAI_API_KEY is not configured"):
        OpenAIProvider(60).chat([], "m", SYSTEM_PROMPT)


def test_empty_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    client.responses._text = "  "
    provider = _provider(monkeypatch, client)
    with pytest.raises(ChatError, match="the provider returned an invalid response"):
        provider.chat([], "m", SYSTEM_PROMPT)


class _HTTPRequest:
    pass


class _HTTPHeaders:
    def get(self, _name: str) -> None:
        return None


class _HTTPResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.request = _HTTPRequest()
        self.headers = _HTTPHeaders()


def _status(cls: type[APIStatusError], status: int, message: str) -> APIStatusError:
    response = cast(Any, _HTTPResponse(status))
    return cls(message, response=response, body=None)


@pytest.mark.parametrize(
    ("exc", "match"),
    [
        (_status(AuthenticationError, 401, "nope"), "OpenAI authentication failed"),
        (_status(PermissionDeniedError, 403, "nope"), "OpenAI authentication failed"),
        (_status(RateLimitError, 429, "nope"), "OpenAI rate limit reached"),
        (
            APITimeoutError(cast(Any, _HTTPRequest())),
            "OpenAI request timed out",
        ),
        (_status(NotFoundError, 404, "missing"), "model 'some-model' is not available"),
        (_status(InternalServerError, 500, "down"), "OpenAI is unavailable"),
    ],
)
def test_catalogue_errors(monkeypatch: pytest.MonkeyPatch, exc: Exception, match: str) -> None:
    client = _Client()
    client.responses._error = exc
    provider = _provider(monkeypatch, client)
    with pytest.raises(ChatError, match=match):
        provider.chat([], "some-model", SYSTEM_PROMPT)


def test_context_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    client.responses._error = _status(BadRequestError, 400, "maximum context length exceeded")
    provider = _provider(monkeypatch, client)
    with pytest.raises(ChatError, match="the conversation is too long for this model"):
        provider.chat([], "m", SYSTEM_PROMPT)


def test_non_sdk_error_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    client.responses._error = ValueError("boom\nsecond line")
    provider = _provider(monkeypatch, client)
    with pytest.raises(ChatError, match="boom") as exc_info:
        provider.chat([], "some-model", SYSTEM_PROMPT)
    assert "second line" not in str(exc_info.value)


def test_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    client.responses._error = APIConnectionError(request=cast(Any, _HTTPRequest()))
    provider = _provider(monkeypatch, client)
    with pytest.raises(ChatError, match="network error talking to OpenAI"):
        provider.chat([], "m", SYSTEM_PROMPT)
