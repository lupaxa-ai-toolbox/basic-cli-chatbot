"""Tests for the YAML profile file."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lupaxa.basic_cli_chatbot.config import (
    ConfigError,
    default_config_path,
    find_profile,
    load_config,
)

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


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_grok_profile_is_accepted(tmp_path: Path) -> None:
    path = _write(tmp_path, "default:\n  provider: grok\n  model: grok-4.6\n")
    profile = find_profile(load_config(path), "default")
    assert profile.provider == "grok"
    assert profile.model == "grok-4.6"


def test_loads_profiles_and_leaves_optional_keys_unset(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "default:\n  provider: OpenAI\n  model: ' gpt-5.6-luna '\n"
        "code-review:\n  provider: ollama\n  model: llama3.2\n"
        "  ollama-host: ' http://127.0.0.1:11434/ '\n"
        "  max-context-messages: 20\n  stream: true\n",
    )
    loaded = load_config(path)
    assert loaded.path == path
    assert loaded.profiles[0].name == "default"
    assert loaded.profiles[0].provider == "openai"
    assert loaded.profiles[0].model == "gpt-5.6-luna"
    assert loaded.profiles[0].ollama_host is None
    assert loaded.profiles[0].max_context_messages is None
    assert loaded.profiles[0].stream is None
    assert loaded.profiles[0].timeout is None
    review = find_profile(loaded, "Code-Review")
    assert review.name == "code-review"
    assert review.provider == "ollama"
    assert review.model == "llama3.2"
    assert review.ollama_host == "http://127.0.0.1:11434/"
    assert review.max_context_messages == 20
    assert review.stream is True


def test_timeout_is_stored_as_a_float(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "default:\n  provider: openai\n  model: m\n  timeout: 30\n"
        "other:\n  provider: ollama\n  model: m\n  timeout: 30.5\n",
    )
    loaded = load_config(path)
    assert loaded.profiles[0].timeout == 30.0
    assert isinstance(loaded.profiles[0].timeout, float)
    assert loaded.profiles[1].timeout == 30.5


def test_timeout_problems(tmp_path: Path) -> None:
    message = "profile 'draft' timeout must be a number of seconds greater than 0"
    cases = (
        "draft:\n  provider: openai\n  model: m\n  timeout: 0\n",
        "draft:\n  provider: openai\n  model: m\n  timeout: -1\n",
        "draft:\n  provider: openai\n  model: m\n  timeout: true\n",
        "draft:\n  provider: openai\n  model: m\n  timeout: yes\n",
        "draft:\n  provider: openai\n  model: m\n  timeout: '30'\n",
        "draft:\n  provider: openai\n  model: m\n  timeout:\n",
        "draft:\n  provider: openai\n  model: m\n  timeout: .inf\n",
        "draft:\n  provider: openai\n  model: m\n  timeout: .nan\n",
    )
    for text in cases:
        with pytest.raises(ConfigError, match=f"^{re.escape(message)}$"):
            load_config(_write(tmp_path, text))


def test_timeout_integer_too_large_to_convert(tmp_path: Path) -> None:
    huge = "9" * 400
    text = f"draft:\n  provider: openai\n  model: m\n  timeout: {huge}\n"
    message = "profile 'draft' timeout must be a number of seconds greater than 0"
    with pytest.raises(ConfigError, match=f"^{re.escape(message)}$"):
        load_config(_write(tmp_path, text))


def test_unknown_key_is_reported_before_a_bad_timeout(tmp_path: Path) -> None:
    text = "draft:\n  provider: openai\n  model: m\n  extra: 1\n  timeout: 0\n"
    with pytest.raises(ConfigError, match="^profile 'draft' has unknown key 'extra'$"):
        load_config(_write(tmp_path, text))


def test_empty_file_has_no_profiles(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, "")).profiles == ()


def test_null_document_has_no_profiles(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, "null\n")).profiles == ()


def test_non_mapping_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="^config file must be a mapping$"):
        load_config(_write(tmp_path, "- a\n- b\n"))


def test_profile_value_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="^profile 'draft' must be a mapping$"):
        load_config(_write(tmp_path, "draft: openai\n"))


def test_second_document(tmp_path: Path) -> None:
    text = "default:\n  provider: openai\n  model: m\n---\nother:\n  provider: ollama\n  model: x\n"
    with pytest.raises(ConfigError, match="^config file must be a single document$"):
        load_config(_write(tmp_path, text))


def test_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="^invalid config file: "):
        load_config(_write(tmp_path, "[\n"))


def test_non_string_profile_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="^profile names must be strings$"):
        load_config(_write(tmp_path, "1:\n  provider: openai\n  model: m\n"))


def test_unhashable_profile_name(tmp_path: Path) -> None:
    text = "? [a, b]\n: {provider: openai, model: m}\n"
    with pytest.raises(ConfigError, match="^profile names must be strings$"):
        load_config(_write(tmp_path, text))


def test_blank_profile_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="^profile name is empty$"):
        load_config(_write(tmp_path, "'  ':\n  provider: openai\n  model: m\n"))


def test_duplicate_profile_ignores_case(tmp_path: Path) -> None:
    text = (
        "Code-Review:\n  provider: openai\n  model: m\n"
        "code-review:\n  provider: gemini\n  model: n\n"
    )
    with pytest.raises(ConfigError, match="^duplicate profile 'code-review'$"):
        load_config(_write(tmp_path, text))


def test_repeated_yaml_key(tmp_path: Path) -> None:
    text = (
        "code-review:\n  provider: openai\n  model: m\n"
        "code-review:\n  provider: gemini\n  model: n\n"
    )
    with pytest.raises(ConfigError, match="^duplicate profile 'code-review'$"):
        load_config(_write(tmp_path, text))


def test_unknown_key_is_reported_before_a_missing_model(tmp_path: Path) -> None:
    text = "draft:\n  provider: openai\n  api-key: secret\n"
    with pytest.raises(ConfigError, match="^profile 'draft' has unknown key 'api-key'$"):
        load_config(_write(tmp_path, text))


def test_provider_and_model_problems(tmp_path: Path) -> None:
    cases = (
        ("draft:\n  model: m\n", "profile 'draft' is missing provider"),
        ("draft:\n  provider: '  '\n  model: m\n", "profile 'draft' is missing provider"),
        ("draft:\n  provider: 1\n  model: m\n", "profile 'draft' provider must be a string"),
        (
            "draft:\n  provider: claude\n  model: m\n",
            "profile 'draft' has unknown provider 'claude' (openai, gemini, grok, ollama)",
        ),
        ("draft:\n  provider: openai\n", "profile 'draft' is missing model"),
        ("draft:\n  provider: openai\n  model: [a]\n", "profile 'draft' model must be a string"),
    )
    for text, message in cases:
        with pytest.raises(ConfigError, match=f"^{re.escape(message)}$"):
            load_config(_write(tmp_path, text))


def test_optional_field_problems(tmp_path: Path) -> None:
    cases = (
        (
            "draft:\n  provider: openai\n  model: m\n  ollama-host: ' '\n",
            "profile 'draft' ollama-host must be a non-empty string",
        ),
        (
            "draft:\n  provider: openai\n  model: m\n  max-context-messages: true\n",
            "profile 'draft' max-context-messages must be an integer",
        ),
        (
            "draft:\n  provider: openai\n  model: m\n  max-context-messages: -1\n",
            "profile 'draft' max-context-messages must be 0 or greater",
        ),
        (
            "draft:\n  provider: openai\n  model: m\n  stream: 'yes'\n",
            "profile 'draft' stream must be true or false",
        ),
    )
    for text, message in cases:
        with pytest.raises(ConfigError, match=f"^{re.escape(message)}$"):
            load_config(_write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match=f"^config file not found: {path}$"):
        load_config(path)


def test_unreadable_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _write(tmp_path, "default:\n  provider: openai\n  model: m\n")

    def boom(*_args: object, **_kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", boom)
    with pytest.raises(ConfigError, match=f"^unable to read config file: {path}$"):
        load_config(path)


def test_unknown_profile_lists_file_spellings(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "Default:\n  provider: openai\n  model: m\nDraft:\n  provider: ollama\n  model: x\n",
    )
    loaded = load_config(path)
    with pytest.raises(ConfigError, match=r"^unknown profile 'nope' \(Default, Draft\)$"):
        find_profile(loaded, " nope ")


def test_blank_lookup_name(tmp_path: Path) -> None:
    loaded = load_config(_write(tmp_path, "default:\n  provider: openai\n  model: m\n"))
    with pytest.raises(ConfigError, match="^profile is required$"):
        find_profile(loaded, "  ")


def test_default_path_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_config_path() == tmp_path / "basic-cli-chatbot" / "config.yaml"


def test_default_path_uses_home_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert default_config_path() == tmp_path / ".config" / "basic-cli-chatbot" / "config.yaml"
