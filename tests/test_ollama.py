"""Tests for the Ollama provider. The official client is faked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from ollama import ResponseError

from lupaxa.basic_cli_chatbot.chat import SYSTEM_PROMPT
from lupaxa.basic_cli_chatbot.providers.errors import ChatError
from lupaxa.basic_cli_chatbot.providers.ollama import OllamaProvider

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


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _ChatResponse:
    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _NoMessage:
    pass


class _Listed:
    def __init__(self, model: str | None) -> None:
        self.model = model


class _ListResponse:
    def __init__(self, models: list[_Listed]) -> None:
        self.models = models


class _FakeClient:
    outcome: dict[str, Any] = {}
    latest: _FakeClient

    def __init__(self, *, host: str, timeout: httpx.Timeout) -> None:
        self.host = host
        self.timeout = timeout
        self.closed = False
        self.calls: list[dict[str, object]] = []
        _FakeClient.latest = self

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = _FakeClient.outcome
        if kwargs.get("stream"):
            chunks = outcome["chunks"]
            if isinstance(chunks, BaseException):
                raise chunks
            return iter(chunks)
        sample = outcome["sample"]
        if isinstance(sample, BaseException):
            raise sample
        return sample

    def list(self) -> object:
        listed = _FakeClient.outcome["listed"]
        if isinstance(listed, BaseException):
            raise listed
        return listed

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch: pytest.MonkeyPatch, outcome: dict[str, Any]) -> None:
    _FakeClient.outcome = outcome

    def factory(*, host: str, timeout: httpx.Timeout) -> _FakeClient:
        return _FakeClient(host=host, timeout=timeout)

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.providers.ollama.Client", factory)


def test_module_does_not_name_hosted_sdks() -> None:
    import lupaxa.basic_cli_chatbot.providers.ollama as ollama

    text = Path(ollama.__file__).read_text(encoding="utf-8")
    assert "openai" not in text
    assert "google" not in text
    assert "xai_sdk" not in text


def _expect_timeout(
    timeout: object, connect: float, read: float, write: float, pool: float
) -> None:
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == connect
    assert timeout.read == read
    assert timeout.write == write
    assert timeout.pool == pool


def test_chat_sends_system_message_then_history(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": _ChatResponse("four")})
    provider = OllamaProvider("http://localhost:11434/", timeout=300)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    assert provider.chat(messages, "llama", SYSTEM_PROMPT) == "four"
    client = _FakeClient.latest
    assert client.host == "http://localhost:11434"
    _expect_timeout(client.timeout, 5.0, 300.0, 30.0, 5.0)
    assert client.closed is True
    call = client.calls[0]
    assert call["model"] == "llama"
    assert call["stream"] is False
    assert call["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 2+2?"},
    ]


def test_timeout_of_ten_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": _ChatResponse("four")})
    provider = OllamaProvider("http://localhost:11434", timeout=10)
    assert provider.chat([], "llama", SYSTEM_PROMPT) == "four"
    _expect_timeout(_FakeClient.latest.timeout, 5.0, 10.0, 10.0, 5.0)


def test_timeout_of_two_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": _ChatResponse("four")})
    provider = OllamaProvider("http://localhost:11434", timeout=2)
    assert provider.chat([], "llama", SYSTEM_PROMPT) == "four"
    _expect_timeout(_FakeClient.latest.timeout, 2.0, 2.0, 2.0, 2.0)


def test_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": httpx.ReadTimeout("slow")})
    provider = OllamaProvider("http://localhost:11434", timeout=300)
    with pytest.raises(ChatError, match="^Ollama request timed out$"):
        provider.chat([], "llama", SYSTEM_PROMPT)


def test_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": ConnectionError("refused")})
    provider = OllamaProvider("http://127.0.0.1:11434", timeout=300)
    with pytest.raises(ChatError, match="^unable to connect to Ollama at http://127.0.0.1:11434$"):
        provider.chat([], "llama", SYSTEM_PROMPT)
    assert _FakeClient.latest.closed is True


def test_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": ResponseError("missing", 404)})
    provider = OllamaProvider("http://localhost:11434", timeout=300)
    with pytest.raises(ChatError) as exc:
        provider.chat([], "missing", SYSTEM_PROMPT)
    assert str(exc.value) == "Ollama model 'missing' is not installed.\nollama pull missing"


def test_other_response_error_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": ResponseError("boom\nsecond", 500)})
    provider = OllamaProvider("http://localhost:11434", timeout=300)
    with pytest.raises(ChatError, match="^boom") as exc_info:
        provider.chat([], "llama", SYSTEM_PROMPT)
    assert "second" not in str(exc_info.value)


def test_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {"listed": _ListResponse([_Listed("llama3.2:latest"), _Listed("mistral")])},
    )
    provider = OllamaProvider("http://localhost:11434", timeout=300)
    assert provider.list_models() == ["llama3.2:latest", "mistral"]
    assert _FakeClient.latest.closed is True


def test_stream_yields_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {
            "chunks": [
                _ChatResponse("fo"),
                _ChatResponse(""),
                _ChatResponse("ur"),
            ]
        },
    )
    provider = OllamaProvider("http://localhost:11434", timeout=300)
    assert list(provider.stream([{"role": "user", "content": "hi"}], "llama", SYSTEM_PROMPT)) == [
        "fo",
        "ur",
    ]
    call = _FakeClient.latest.calls[0]
    assert call["stream"] is True
    assert _FakeClient.latest.closed is True


def test_unreadable_reply_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"sample": _NoMessage()})
    provider = OllamaProvider("http://localhost:11434", timeout=300)
    with pytest.raises(ChatError, match="^the provider returned an invalid response$"):
        provider.chat([], "llama", SYSTEM_PROMPT)
