"""Tests for the basic-cli-chatbot command."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lupaxa.basic_cli_chatbot.chat import SYSTEM_PROMPT
from lupaxa.basic_cli_chatbot.cli import main
from lupaxa.basic_cli_chatbot.providers.errors import ChatError
from lupaxa.basic_cli_chatbot.version import __version__

REQUIRED = "Error: provider is required (openai, gemini, grok, ollama)\n"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BASIC_CLI_CHATBOT_PROVIDER",
        "BASIC_CLI_CHATBOT_MODEL",
        "BASIC_CLI_CHATBOT_TIMEOUT",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeProvider:
    """In-memory provider. Outcomes are returned or raised in order."""

    mode = "Remote"
    supports_stream = True

    def __init__(self, outcomes: list[str | BaseException]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "system_prompt": system_prompt,
            }
        )
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> Iterator[str]:
        raise ChatError("model discovery is unavailable for openai")

    def list_models(self) -> list[str]:
        return []


def _forbid_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> FakeProvider:
        raise AssertionError("build called")

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.build", boom)


def _patch(monkeypatch: pytest.MonkeyPatch, outcomes: list[str | BaseException]) -> FakeProvider:
    provider = FakeProvider(outcomes)

    def fake_build(name: str, *, ollama_host: str, timeout: float) -> FakeProvider:
        return provider

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.build", fake_build)
    return provider


def _script_input(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    answers = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))


def test_version_does_not_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_build(monkeypatch)
    assert main(["--version"]) == 0
    assert main(["--version", "hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"basic-cli-chatbot {__version__}\nbasic-cli-chatbot {__version__}\n"
    assert "hello" not in captured.out


def test_missing_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_build(monkeypatch)
    assert main([]) == 2
    assert main(["hello"]) == 2
    assert capsys.readouterr().err == REQUIRED + REQUIRED


def test_unknown_provider(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "nope", "--model", "m"]) == 2
    assert "unknown provider 'nope' (openai, gemini, grok, ollama)" in capsys.readouterr().err


def test_missing_model(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _forbid_build(monkeypatch)
    assert main(["--provider", "openai", "hello"]) == 2
    assert capsys.readouterr().err == "Error: model is required\n"


def test_flag_beats_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_build(monkeypatch)
    monkeypatch.setenv("BASIC_CLI_CHATBOT_PROVIDER", "ollama")
    monkeypatch.setenv("BASIC_CLI_CHATBOT_MODEL", "from-env")
    assert main(["--show-config", "--provider", "openai", "--model", "from-flag"]) == 0
    out = capsys.readouterr().out
    assert "provider: openai" in out
    assert "model: from-flag" in out


def test_show_config_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_build(monkeypatch)
    assert main(["--show-config"]) == 0
    out = capsys.readouterr().out
    assert "provider: not set" in out
    assert "model: not set" in out
    assert "mode: not set" in out
    assert "max-context-messages: 0" in out
    assert "ollama-host" not in out
    assert "api-key" not in out
    assert "profile:" not in out
    assert "config:" not in out


def test_show_config_ollama(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--show-config", "--provider", "ollama", "--model", "llama3.2"]) == 0
    assert capsys.readouterr().out == (
        "provider: ollama\n"
        "model: llama3.2\n"
        "mode: Local\n"
        "max-context-messages: 0\n"
        "timeout: 300\n"
        "ollama-host: http://localhost:11434\n"
    )


def test_show_config_hides_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    assert main(["--show-config", "--provider", "openai"]) == 0
    out = capsys.readouterr().out
    assert "api-key: configured" in out
    assert "mode: Remote" in out
    assert "sk-secret" not in out
    monkeypatch.delenv("OPENAI_API_KEY")
    assert main(["--show-config", "--provider", "openai"]) == 0
    assert "api-key: not configured" in capsys.readouterr().out


def test_show_config_gemini_omits_host(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--show-config", "--provider", "gemini", "--ollama-host", "http://example:1"]) == 0
    out = capsys.readouterr().out
    assert "ollama-host" not in out
    assert "api-key: not configured" in out


def test_show_config_grok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    assert main(["--show-config", "--provider", "grok", "--model", "grok-4.6"]) == 0
    out = capsys.readouterr().out
    assert out == (
        "provider: grok\n"
        "model: grok-4.6\n"
        "mode: Remote\n"
        "max-context-messages: 0\n"
        "timeout: 60\n"
        "api-key: configured\n"
    )
    assert "xai-secret" not in out
    assert "ollama-host" not in out


def test_question_with_show_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "openai", "--model", "m", "--show-config", "hello"]) == 2
    assert capsys.readouterr().err == (
        "Error: choose one of a question, --list-models, --show-config, or --validate\n"
    )


def test_negative_context(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "openai", "--max-context-messages", "-1", "--show-config"]) == 2
    assert capsys.readouterr().err == "Error: --max-context-messages must be 0 or greater\n"


def test_oneshot_prints_answer_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = _patch(monkeypatch, ["four"])
    assert (
        main(["--no-stream", "--provider", "openai", "--model", "some-model", "What is 2+2?"]) == 0
    )
    captured = capsys.readouterr()
    assert captured.out == "four\n"
    assert captured.err == ""
    assert provider.calls[0]["model"] == "some-model"
    assert provider.calls[0]["system_prompt"] == SYSTEM_PROMPT
    assert provider.calls[0]["messages"] == [{"role": "user", "content": "What is 2+2?"}]


def test_oneshot_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch(monkeypatch, [ChatError("offline")])
    assert main(["--no-stream", "--provider", "openai", "--model", "m", "hello"]) == 1
    assert capsys.readouterr().err == "Error: offline\n"


def test_empty_question(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "openai", "--model", "m", ""]) == 2
    assert main(["--provider", "openai", "--model", "m", "   "]) == 2
    assert capsys.readouterr().err == "Error: question is empty\nError: question is empty\n"


def test_oneshot_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch(monkeypatch, [KeyboardInterrupt()])
    assert main(["--no-stream", "--provider", "openai", "--model", "m", "hello"]) == 0
    assert capsys.readouterr().out == "\nGoodbye!\n"


def test_extra_positional_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["one", "two"])
    assert exc.value.code == 2


def test_interactive_keeps_history(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = _patch(monkeypatch, ["first", "second"])
    _script_input(monkeypatch, ["hello", "again", "exit"])
    assert main(["--no-stream", "--provider", "openai", "--model", "some-model"]) == 0
    out = capsys.readouterr().out
    assert "Provider: openai" in out
    assert "Model: some-model" in out
    assert "Mode: Remote" in out
    assert "AI: first" in out
    assert "AI: second" in out
    assert out.endswith("Goodbye!\n")
    assert provider.calls[1]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "again"},
    ]


def test_clear_drops_history(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = _patch(monkeypatch, ["first", "second"])
    _script_input(monkeypatch, ["hello", "clear", "again", "quit"])
    assert main(["--no-stream", "--provider", "openai", "--model", "m"]) == 0
    assert "Conversation context cleared." in capsys.readouterr().out
    assert provider.calls[1]["messages"] == [{"role": "user", "content": "again"}]


def test_failed_turn_is_dropped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = _patch(monkeypatch, [ChatError("offline"), "recovered"])
    _script_input(monkeypatch, ["hello", "again", "exit"])
    assert main(["--no-stream", "--provider", "openai", "--model", "m"]) == 0
    assert "Error: offline" in capsys.readouterr().out
    assert provider.calls[1]["messages"] == [{"role": "user", "content": "again"}]


def test_build_failure_retries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = FakeProvider(["ok"])
    calls = {"count": 0}

    def fake_build(name: str, *, ollama_host: str, timeout: float) -> FakeProvider:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("no key")
        return provider

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.build", fake_build)
    _script_input(monkeypatch, ["hello", "again", "exit"])
    assert main(["--no-stream", "--provider", "openai", "--model", "m"]) == 0
    assert "Error: no key" in capsys.readouterr().out
    assert provider.calls[0]["messages"] == [{"role": "user", "content": "again"}]


def test_ollama_banner_is_local(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("hosted client constructed")

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.providers.openai.OpenAI", boom)
    monkeypatch.setattr("lupaxa.basic_cli_chatbot.providers.gemini.genai.Client", boom)
    _script_input(monkeypatch, ["exit"])
    assert main(["--provider", "ollama", "--model", "llama"]) == 0
    assert "Mode: Local" in capsys.readouterr().out


class ListingProvider(FakeProvider):
    def __init__(self, names: list[str] | None = None, error: BaseException | None = None) -> None:
        super().__init__([])
        self._names = names or []
        self._error = error

    def list_models(self) -> list[str]:
        if self._error is not None:
            raise self._error
        return self._names

    def chat(self, messages, model, system_prompt):
        raise AssertionError("chat called")


class DeltaProvider(FakeProvider):
    def __init__(self, deltas: list[str | BaseException], *, supports_stream: bool = True) -> None:
        super().__init__(["full", "full"])
        self.supports_stream = supports_stream
        self._deltas = deltas
        self.stream_calls: list[list[dict[str, str]]] = []
        self._streams = 0

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> Iterator[str]:
        self.stream_calls.append(list(messages))
        self._streams += 1
        for item in self._deltas:
            if isinstance(item, BaseException) and self._streams == 1:
                raise item
            if not isinstance(item, BaseException):
                yield item


def _use(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.cli.build",
        lambda name, *, ollama_host, timeout: provider,
    )


def test_list_models(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _use(monkeypatch, ListingProvider(["llama", "mistral"]))
    assert main(["--list-models", "--provider", "ollama"]) == 0
    assert capsys.readouterr().out == "llama\nmistral\n"


def test_list_models_requires_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_build(monkeypatch)
    assert main(["--list-models"]) == 2
    assert capsys.readouterr().err == REQUIRED


def test_list_models_discovery_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use(
        monkeypatch,
        ListingProvider(error=ChatError("model discovery is unavailable for openai")),
    )
    assert main(["--list-models", "--provider", "openai"]) == 1
    assert capsys.readouterr().err == "Error: model discovery is unavailable for openai\n"


def test_question_and_list_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-models", "--provider", "openai", "hello"]) == 2
    assert "choose one of a question" in capsys.readouterr().err


def test_stream_only_for_chat(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--stream", "--list-models", "--provider", "openai"]) == 2
    assert main(["--stream", "--show-config", "--provider", "openai"]) == 2
    assert capsys.readouterr().err == (
        "Error: --stream applies only to a chat\nError: --stream applies only to a chat\n"
    )


def test_slash_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ListingProvider()
    _use(monkeypatch, provider)
    _script_input(monkeypatch, ["/provider", "/model", "/context", "exit"])
    assert main(["--provider", "openai", "--model", "some-model"]) == 0
    out = capsys.readouterr().out
    assert "provider: openai" in out
    assert "model: some-model" in out
    assert "context: 0 messages, limit unlimited" in out
    assert provider.calls == []


def test_context_count_after_a_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch(monkeypatch, ["four"])
    _script_input(monkeypatch, ["/context", "hello", "/context", "exit"])
    assert (
        main(
            [
                "--no-stream",
                "--provider",
                "openai",
                "--model",
                "m",
                "--max-context-messages",
                "4",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "context: 0 messages, limit 4" in out
    assert "context: 2 messages, limit 4" in out


def test_chat_streams_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = DeltaProvider(["fo", "ur"])
    _use(monkeypatch, provider)
    assert main(["--provider", "openai", "--model", "m", "hello"]) == 0
    assert capsys.readouterr().out == "four\n"
    assert provider.stream_calls == [[{"role": "user", "content": "hello"}]]
    assert provider.calls == []


def test_no_stream_prints_the_whole_reply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = DeltaProvider(["fo", "ur"])
    _use(monkeypatch, provider)
    assert main(["--no-stream", "--provider", "openai", "--model", "m", "hello"]) == 0
    assert capsys.readouterr().out == "full\n"
    assert provider.stream_calls == []
    assert provider.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_profile_can_turn_streaming_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _config(tmp_path, _REVIEW)
    provider = DeltaProvider(["fo", "ur"])
    _use(monkeypatch, provider)
    assert main(["--profile", "code-review", "--config", path, "ping"]) == 0
    assert capsys.readouterr().out == "full\n"
    assert provider.stream_calls == []


def test_stream_joins_deltas(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = DeltaProvider(["fo", "ur"])
    _use(monkeypatch, provider)
    assert main(["--stream", "--provider", "openai", "--model", "m", "hello"]) == 0
    assert capsys.readouterr().out == "four\n"
    _script_input(monkeypatch, ["one", "two", "exit"])
    assert main(["--stream", "--provider", "openai", "--model", "m"]) == 0
    assert provider.stream_calls[2] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "two"},
    ]


def test_stream_fallback_before_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = DeltaProvider([], supports_stream=False)
    _use(monkeypatch, provider)
    _script_input(monkeypatch, ["hello", "exit"])
    assert main(["--stream", "--provider", "openai", "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert "Streaming is unavailable; using a single response." in out
    assert "AI: full" in out
    assert main(["--stream", "--provider", "openai", "--model", "m", "hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "full\n"
    assert "Streaming is unavailable; using a single response." in captured.err


def test_oneshot_stream_failure_after_delta(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = DeltaProvider(["fo", ChatError("dropped")])
    _use(monkeypatch, provider)
    assert main(["--stream", "--provider", "openai", "--model", "m", "hello"]) == 1
    captured = capsys.readouterr()
    assert "fo" in captured.out
    assert captured.err == "Error: dropped\n"


class _FailOnce(FakeProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.supports_stream = True
        self.stream_calls: list[list[dict[str, str]]] = []
        self._n = 0

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> Iterator[str]:
        self.stream_calls.append(list(messages))
        self._n += 1
        if self._n == 1:
            yield "fo"
            raise ChatError("dropped")
        yield "ok"


def test_interactive_stream_failure_drops_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = _FailOnce()
    _use(monkeypatch, provider)
    _script_input(monkeypatch, ["bad", "later", "exit"])
    assert main(["--stream", "--provider", "openai", "--model", "m"]) == 0
    assert provider.stream_calls[1] == [{"role": "user", "content": "later"}]


def test_ollama_oneshot_skips_hosted_sdks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("hosted client constructed")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gk-secret")
    monkeypatch.setattr("lupaxa.basic_cli_chatbot.providers.openai.OpenAI", boom)
    monkeypatch.setattr("lupaxa.basic_cli_chatbot.providers.gemini.genai.Client", boom)
    _patch(monkeypatch, ["four"])
    assert main(["--no-stream", "--provider", "ollama", "--model", "llama", "hello"]) == 0
    assert capsys.readouterr().out == "four\n"


def _config(tmp_path: Path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


_REVIEW = (
    "default:\n"
    "  provider: openai\n"
    "  model: from-default\n"
    "  max-context-messages: 20\n"
    "  stream: true\n"
    "code-review:\n"
    "  provider: ollama\n"
    "  model: llama3.2\n"
    "  ollama-host: http://127.0.0.1:11434\n"
    "  max-context-messages: 4\n"
    "  stream: false\n"
)


def test_version_does_not_read_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_path: object) -> object:
        raise AssertionError("config read")

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.load_config", boom)
    assert main(["--version"]) == 0
    assert "config read" not in capsys.readouterr().err


def test_profile_supplies_provider_and_ignores_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("BASIC_CLI_CHATBOT_PROVIDER", "gemini")
    monkeypatch.setenv("BASIC_CLI_CHATBOT_MODEL", "from-env")
    path = _config(tmp_path, _REVIEW)
    assert main(["--show-config", "--profile", "Code-Review", "--config", path]) == 0
    out = capsys.readouterr().out
    assert "provider: ollama\n" in out
    assert "model: llama3.2\n" in out
    assert "max-context-messages: 4\n" in out
    assert "ollama-host: http://127.0.0.1:11434\n" in out
    assert "profile: code-review\n" in out
    assert f"config: {path}\n" in out
    assert "from-env" not in out
    assert "sk-" not in out


def test_profile_rejects_provider_and_model(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _config(tmp_path, _REVIEW)
    assert main(["--profile", "code-review", "--provider", "openai", "--config", path]) == 2
    assert main(["--profile", "code-review", "--model", "m", "--config", path]) == 2
    err = capsys.readouterr().err
    message = "Error: --profile cannot be combined with --provider or --model\n"
    assert err == message + message


def test_blank_profile_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--profile", "  "]) == 2
    assert capsys.readouterr().err == "Error: profile is required\n"


def test_default_profile_beats_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("BASIC_CLI_CHATBOT_PROVIDER", "gemini")
    monkeypatch.setenv("BASIC_CLI_CHATBOT_MODEL", "from-env")
    path = _config(tmp_path, _REVIEW)
    assert main(["--show-config", "--config", path]) == 0
    out = capsys.readouterr().out
    assert "provider: openai\n" in out
    assert "model: from-default\n" in out
    assert "max-context-messages: 20\n" in out
    assert "profile: default\n" in out
    assert "ollama-host" not in out


def test_one_flag_does_not_open_the_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def boom(_path: object) -> object:
        raise AssertionError("config read")

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.load_config", boom)
    path = _config(tmp_path, _REVIEW)
    assert main(["--provider", "openai", "--config", path]) == 2
    assert capsys.readouterr().err == "Error: model is required\n"


def test_omitted_optional_flag_keeps_the_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _config(tmp_path, _REVIEW)
    assert main(["--show-config", "--profile", "code-review", "--config", path]) == 0
    out = capsys.readouterr().out
    assert "max-context-messages: 4\n" in out
    assert "ollama-host: http://127.0.0.1:11434\n" in out


def test_passed_optional_flags_replace_the_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _config(tmp_path, _REVIEW)
    assert (
        main(
            [
                "--show-config",
                "--profile",
                "code-review",
                "--config",
                path,
                "--ollama-host",
                "http://10.0.0.2:11434",
                "--max-context-messages",
                "9",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "ollama-host: http://10.0.0.2:11434\n" in out
    assert "max-context-messages: 9\n" in out


def test_default_profile_streams(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _config(tmp_path, _REVIEW)
    provider = _patch(monkeypatch, ["four"])

    def stream(
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ):
        del messages, model, system_prompt
        yield "fo"
        yield "ur"

    provider.stream = stream  # type: ignore[method-assign]
    assert main(["--profile", "default", "--config", path, "ping"]) == 0
    assert capsys.readouterr().out == "four\n"
    assert provider.calls == []


def test_no_stream_uses_chat(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _config(tmp_path, _REVIEW)
    provider = _patch(monkeypatch, ["four"])
    assert main(["--profile", "default", "--config", path, "--no-stream", "ping"]) == 0
    assert capsys.readouterr().out == "four\n"
    assert provider.calls[0]["model"] == "from-default"


def test_missing_default_path_keeps_provider_required(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2
    assert capsys.readouterr().err == REQUIRED


def test_missing_explicit_config(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert main(["--validate", "--config", str(missing)]) == 2
    assert main(["--profile", "default", "--config", str(missing)]) == 2
    err = capsys.readouterr().err
    message = f"Error: config file not found: {missing}\n"
    assert err == message + message


def test_validate_ok(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _config(tmp_path, _REVIEW)
    assert main(["--validate", "--config", path]) == 0
    assert capsys.readouterr().out == "config: ok\nprofiles: default, code-review\n"


def test_validate_empty(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _config(tmp_path, "")
    assert main(["--validate", "--config", path]) == 0
    assert capsys.readouterr().out == "config: ok\nprofiles:\n"


def test_validate_first_error(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _config(tmp_path, "draft:\n  provider: openai\n  api-key: secret\n")
    assert main(["--validate", "--config", path]) == 2
    assert capsys.readouterr().err == "Error: profile 'draft' has unknown key 'api-key'\n"


def test_validate_rejects_other_actions(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _config(tmp_path, _REVIEW)
    choose = "Error: choose one of a question, --list-models, --show-config, or --validate\n"
    combo = "Error: --validate cannot be combined with --profile, --provider, or --model\n"
    assert main(["--validate", "hello", "--config", path]) == 2
    assert main(["--validate", "--list-models", "--config", path]) == 2
    assert main(["--validate", "--show-config", "--config", path]) == 2
    assert main(["--validate", "--profile", "default", "--config", path]) == 2
    assert main(["--validate", "--provider", "openai", "--config", path]) == 2
    assert main(["--validate", "--model", "m", "--config", path]) == 2
    err = capsys.readouterr().err
    assert err == choose + choose + choose + combo + combo + combo


def test_stream_flags_reject_non_chat(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _config(tmp_path, _REVIEW)
    message = "Error: --stream applies only to a chat\n"
    assert main(["--validate", "--stream", "--config", path]) == 2
    assert main(["--validate", "--no-stream", "--config", path]) == 2
    assert capsys.readouterr().err == message + message


def test_both_stream_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--stream", "--no-stream", "--provider", "openai", "--model", "m"]) == 2
    assert capsys.readouterr().err == "Error: --stream cannot be combined with --no-stream\n"


def test_list_models_uses_profile_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _config(tmp_path, _REVIEW)
    seen: dict[str, str] = {}

    def fake_build(name: str, *, ollama_host: str, timeout: float) -> FakeProvider:
        seen["name"] = name
        seen["host"] = ollama_host
        provider = FakeProvider([])
        provider.list_models = lambda: ["llama3.2"]  # type: ignore[method-assign]
        return provider

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.build", fake_build)
    assert main(["--list-models", "--profile", "code-review", "--config", path]) == 0
    assert seen == {"name": "ollama", "host": "http://127.0.0.1:11434"}
    assert capsys.readouterr().out == "llama3.2\n"


def test_unknown_profile(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _config(tmp_path, _REVIEW)
    assert main(["--profile", "nope", "--config", path]) == 2
    assert capsys.readouterr().err == "Error: unknown profile 'nope' (default, code-review)\n"


def test_show_config_timeout_not_set(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--show-config"]) == 0
    out = capsys.readouterr().out
    assert "max-context-messages: 0\ntimeout: not set\n" in out


def test_show_config_prints_catalog_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--show-config", "--provider", "openai"]) == 0
    assert "max-context-messages: 0\ntimeout: 60\n" in capsys.readouterr().out
    assert main(["--show-config", "--provider", "ollama"]) == 0
    assert "max-context-messages: 0\ntimeout: 300\nollama-host:" in capsys.readouterr().out


def test_show_config_prints_a_decimal_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--show-config", "--timeout", "30.5"]) == 0
    captured = capsys.readouterr()
    assert "timeout: 30.5\n" in captured.out
    assert "60.0" not in captured.out


def test_timeout_flag_beats_profile_and_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", "9")
    path = _config(
        tmp_path,
        "default:\n  provider: openai\n  model: m\n  timeout: 15\n",
    )
    assert main(["--show-config", "--config", path, "--timeout", "30"]) == 0
    assert "timeout: 30\n" in capsys.readouterr().out


def test_profile_timeout_beats_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", "nope")
    path = _config(
        tmp_path,
        "default:\n  provider: openai\n  model: m\n  timeout: 15\n",
    )
    assert main(["--show-config", "--config", path]) == 0
    captured = capsys.readouterr()
    assert "timeout: 15\n" in captured.out
    assert "nope" not in captured.err


def test_env_timeout_fills_an_omitted_profile_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", " 12.5 ")
    path = _config(tmp_path, "default:\n  provider: openai\n  model: m\n")
    assert main(["--show-config", "--config", path]) == 0
    assert "timeout: 12.5\n" in capsys.readouterr().out


def test_bad_timeout_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["--timeout", "0", "--show-config"]) == 2
    assert main(["--timeout", "-1", "--provider", "openai"]) == 2
    assert main(["--timeout", "inf", "--show-config"]) == 2
    flag = "Error: --timeout must be a number of seconds greater than 0\n"
    assert capsys.readouterr().err == flag + flag + flag
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", "nope")
    assert main(["--show-config", "--provider", "openai"]) == 2
    env = "Error: BASIC_CLI_CHATBOT_TIMEOUT must be a number of seconds greater than 0\n"
    assert capsys.readouterr().err == env


def test_bad_env_does_not_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_build(monkeypatch)
    message = "Error: BASIC_CLI_CHATBOT_TIMEOUT must be a number of seconds greater than 0\n"
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", "0")
    assert main(["--provider", "openai", "--model", "m", "hi"]) == 2
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", "   ")
    assert main(["--provider", "openai", "--model", "m", "hi"]) == 2
    assert capsys.readouterr().err == message + message


def test_bad_env_is_ignored_when_the_flag_is_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", "nope")
    assert main(["--show-config", "--timeout", "8"]) == 0
    assert "timeout: 8\n" in capsys.readouterr().out


def test_gemini_floor_runs_before_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_build(monkeypatch)
    assert main(["--list-models", "--provider", "gemini", "--timeout", "0.0004"]) == 2
    assert capsys.readouterr().err == ("Error: timeout must be at least 1 millisecond for Gemini\n")


def test_gemini_accepts_one_millisecond(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, float] = {}

    def fake_build(name: str, *, ollama_host: str, timeout: float) -> FakeProvider:
        seen["timeout"] = timeout
        provider = FakeProvider([])
        provider.list_models = lambda: ["gemini-a"]  # type: ignore[method-assign]
        return provider

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.build", fake_build)
    assert main(["--list-models", "--provider", "gemini", "--timeout", "0.0006"]) == 0
    assert seen["timeout"] == 0.0006
    assert capsys.readouterr().out == "gemini-a\n"


def test_catalog_default_reaches_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("BASIC_CLI_CHATBOT_TIMEOUT", raising=False)
    seen: dict[str, float] = {}

    def fake_build(name: str, *, ollama_host: str, timeout: float) -> FakeProvider:
        seen[name] = timeout
        provider = FakeProvider([])
        provider.list_models = lambda: [name]  # type: ignore[method-assign]
        return provider

    monkeypatch.setattr("lupaxa.basic_cli_chatbot.cli.build", fake_build)
    assert main(["--list-models", "--provider", "openai"]) == 0
    assert main(["--list-models", "--provider", "ollama"]) == 0
    assert seen == {"openai": 60.0, "ollama": 300.0}
    assert capsys.readouterr().out == "openai\nollama\n"


def test_timeout_zero_blocks_validate(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _config(tmp_path, "default:\n  provider: openai\n  model: m\n")
    assert main(["--validate", "--timeout", "0", "--config", path]) == 2
    captured = capsys.readouterr()
    assert captured.err == ("Error: --timeout must be a number of seconds greater than 0\n")
    assert "config: ok" not in captured.out


def test_valid_timeout_still_validates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("BASIC_CLI_CHATBOT_TIMEOUT", "nope")
    path = _config(tmp_path, "default:\n  provider: openai\n  model: m\n  timeout: 10\n")
    assert main(["--validate", "--timeout", "3", "--config", path]) == 0
    assert capsys.readouterr().out == "config: ok\nprofiles: default\n"


def test_missing_model_wins_over_the_gemini_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--provider", "gemini", "--timeout", "0.0004"]) == 2
    assert capsys.readouterr().err == "Error: model is required\n"
