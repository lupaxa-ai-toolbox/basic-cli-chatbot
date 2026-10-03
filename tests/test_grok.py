"""Tests for the Grok provider. The SDK client is faked."""

from __future__ import annotations

from typing import Any

import grpc
import pytest

from lupaxa.basic_cli_chatbot.chat import SYSTEM_PROMPT
from lupaxa.basic_cli_chatbot.providers.errors import ChatError
from lupaxa.basic_cli_chatbot.providers.grok import GrokProvider

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


class _RpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def __str__(self) -> str:
        return self._details


class _Chunk:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _NoContent:
    pass


class _Chat:
    def __init__(self, sample: object, chunks: list[object] | None = None) -> None:
        self._sample = sample
        self._chunks = chunks or []

    def sample(self) -> object:
        if isinstance(self._sample, BaseException):
            raise self._sample
        return self._sample

    def stream(self):
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield object(), chunk


class _Models:
    def __init__(self, models: list[object] | BaseException) -> None:
        self._models = models

    def list_language_models(self) -> list[object]:
        if isinstance(self._models, BaseException):
            raise self._models
        return self._models


class _FakeClient:
    outcome: dict[str, Any] = {}

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False
        outcome = _FakeClient.outcome
        self.chat_api = _ChatApi(outcome)
        self.models = _Models(outcome.get("models", []))

    @property
    def chat(self) -> _ChatApi:
        return self.chat_api

    def close(self) -> None:
        self.closed = True


class _ChatApi:
    def __init__(self, outcome: dict[str, Any]) -> None:
        self.outcome = outcome
        self.created: dict[str, object] = {}

    def create(self, **kwargs: object) -> _Chat:
        self.created = kwargs
        return _Chat(self.outcome.get("sample", _Content("four")), self.outcome.get("chunks"))


class _Content:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Named:
    def __init__(self, name: str | None) -> None:
        self.name = name


def _install(monkeypatch: pytest.MonkeyPatch, outcome: dict[str, Any]) -> list[_FakeClient]:
    created: list[_FakeClient] = []
    _FakeClient.outcome = outcome

    def factory(**kwargs: object) -> _FakeClient:
        client = _FakeClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.providers.grok.Client", factory)
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.grok.system",
        lambda text: ("system", text),
    )
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.grok.user",
        lambda text: ("user", text),
    )
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.grok.assistant",
        lambda text: ("assistant", text),
    )
    return created


def test_module_does_not_name_other_sdks() -> None:
    from pathlib import Path

    import lupaxa.basic_cli_chatbot.providers.grok as grok

    text = Path(grok.__file__).read_text(encoding="utf-8")
    assert "openai" not in text
    assert "google" not in text
    assert "ollama" not in text


def test_chat_sends_system_then_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    created = _install(monkeypatch, {"sample": _Content("four")})
    provider = GrokProvider(30.5)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Thanks"},
    ]
    assert provider.chat(messages, "grok-4.6", SYSTEM_PROMPT) == "four"
    client = created[0]
    assert client.kwargs["api_key"] == "xai-test"
    assert client.kwargs["timeout"] == 30.5
    assert client.kwargs["channel_options"] == [("grpc.enable_retries", 0)]
    assert client.closed is True
    sent = client.chat_api.created
    assert sent["model"] == "grok-4.6"
    assert sent["store_messages"] is False
    assert "previous_response_id" not in sent
    assert sent["messages"] == [
        ("system", SYSTEM_PROMPT),
        ("user", "What is 2+2?"),
        ("assistant", "4"),
        ("user", "Thanks"),
    ]


def test_blank_key_does_not_construct_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "   ")
    created = _install(monkeypatch, {})
    provider = GrokProvider(60)
    with pytest.raises(ChatError, match="^XAI_API_KEY is not configured$"):
        provider.chat([], "grok-4.6", SYSTEM_PROMPT)
    assert created == []


def test_stream_yields_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    created = _install(
        monkeypatch,
        {"chunks": [_Chunk("fo"), _Chunk(""), _Chunk("ur")]},
    )
    provider = GrokProvider(60)
    assert list(
        provider.stream([{"role": "user", "content": "hi"}], "grok-4.6", SYSTEM_PROMPT)
    ) == [
        "fo",
        "ur",
    ]
    assert created[0].closed is True


def test_list_models_returns_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    created = _install(
        monkeypatch,
        {"models": [_Named("grok-4.6"), _Named("grok-4")]},
    )
    provider = GrokProvider(60)
    assert provider.list_models() == ["grok-4.6", "grok-4"]
    assert created[0].closed is True


def test_unknown_role_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    created = _install(monkeypatch, {})
    provider = GrokProvider(60)
    with pytest.raises(ChatError, match="^the provider returned an invalid response$"):
        provider.chat([{"role": "tool", "content": "nope"}], "grok-4.6", SYSTEM_PROMPT)
    assert created == []


@pytest.mark.parametrize(
    ("code", "details", "expected"),
    [
        (grpc.StatusCode.UNAUTHENTICATED, "bad", "Grok authentication failed"),
        (grpc.StatusCode.PERMISSION_DENIED, "bad", "Grok authentication failed"),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, "bad", "Grok rate limit reached"),
        (grpc.StatusCode.DEADLINE_EXCEEDED, "bad", "Grok request timed out"),
        (grpc.StatusCode.NOT_FOUND, "bad", "model 'grok-4.6' is not available"),
        (
            grpc.StatusCode.INVALID_ARGUMENT,
            "maximum context length exceeded",
            "the conversation is too long for this model",
        ),
        (grpc.StatusCode.UNAVAILABLE, "bad", "Grok is unavailable"),
        (grpc.StatusCode.INTERNAL, "bad", "Grok is unavailable"),
    ],
)
def test_status_maps_to_the_grok_message(
    monkeypatch: pytest.MonkeyPatch,
    code: grpc.StatusCode,
    details: str,
    expected: str,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    _install(monkeypatch, {"sample": _RpcError(code, details)})
    provider = GrokProvider(60)
    with pytest.raises(ChatError, match=f"^{expected}$"):
        provider.chat([], "grok-4.6", SYSTEM_PROMPT)


def test_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    _install(monkeypatch, {"sample": ConnectionError("reset")})
    provider = GrokProvider(60)
    with pytest.raises(ChatError, match="^network error talking to Grok$"):
        provider.chat([], "grok-4.6", SYSTEM_PROMPT)


def test_other_error_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    _install(monkeypatch, {"sample": RuntimeError("boom\nsecond")})
    provider = GrokProvider(60)
    with pytest.raises(ChatError, match="^boom$") as exc_info:
        provider.chat([], "grok-4.6", SYSTEM_PROMPT)
    assert "second" not in str(exc_info.value)


def test_blank_reply_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    _install(monkeypatch, {"sample": _Content("  ")})
    provider = GrokProvider(60)
    with pytest.raises(ChatError, match="^the provider returned an invalid response$"):
        provider.chat([], "grok-4.6", SYSTEM_PROMPT)


def test_unreadable_chunk_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    _install(monkeypatch, {"chunks": [_NoContent()]})
    provider = GrokProvider(60)
    with pytest.raises(ChatError, match="^the provider returned an invalid response$"):
        list(provider.stream([], "grok-4.6", SYSTEM_PROMPT))
