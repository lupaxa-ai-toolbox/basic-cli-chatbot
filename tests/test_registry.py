"""Tests for provider selection."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lupaxa.basic_cli_chatbot.providers import KNOWN, build
from lupaxa.basic_cli_chatbot.providers.errors import ChatError
from lupaxa.basic_cli_chatbot.providers.gemini import GeminiProvider
from lupaxa.basic_cli_chatbot.providers.ollama import OllamaProvider
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


def test_known_order() -> None:
    assert KNOWN == ("openai", "gemini", "grok", "ollama")


def test_build_casefolds(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Sentinel:
        pass

    monkeypatch.setattr(
        "lupaxa.basic_cli_chatbot.providers.openai.OpenAIProvider",
        lambda timeout: _Sentinel(),
    )
    built = build("OpenAI", ollama_host="http://localhost:11434", timeout=60)
    assert isinstance(built, _Sentinel)


def test_build_ollama_passes_host() -> None:
    provider = build("ollama", ollama_host="http://localhost:11434/", timeout=60)
    assert isinstance(provider, OllamaProvider)
    assert provider.host == "http://localhost:11434"


def test_build_gemini_and_openai_types() -> None:
    assert isinstance(
        build("gemini", ollama_host="http://localhost:11434", timeout=60), GeminiProvider
    )
    assert isinstance(
        build("openai", ollama_host="http://localhost:11434", timeout=60), OpenAIProvider
    )


def test_unknown_provider() -> None:
    with pytest.raises(
        ChatError,
        match=r"unknown provider 'nope' \(openai, gemini, grok, ollama\)",
    ):
        build("Nope", ollama_host="http://localhost:11434", timeout=60)


def test_ollama_build_does_not_import_hosted_sdks() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    code = "\n".join(
        [
            "import sys",
            "from lupaxa.basic_cli_chatbot.providers import build",
            "build('ollama', ollama_host='http://localhost:11434', timeout=300)",
            "assert 'openai' not in sys.modules",
            "assert 'google' not in sys.modules",
            "assert 'xai_sdk' not in sys.modules",
        ]
    )
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src) if not previous else os.pathsep.join((str(src), previous))
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_build_grok_type() -> None:
    from lupaxa.basic_cli_chatbot.providers.grok import GrokProvider

    built = build("grok", ollama_host="http://localhost:11434", timeout=60)
    assert isinstance(built, GrokProvider)
    assert built.timeout == 60


def test_grok_build_does_not_import_other_sdks() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    code = "\n".join(
        [
            "import sys",
            "from lupaxa.basic_cli_chatbot.providers import build",
            "build('grok', ollama_host='http://localhost:11434', timeout=60)",
            "assert 'openai' not in sys.modules",
            "assert 'google.genai' not in sys.modules",
            "assert 'ollama' not in sys.modules",
        ]
    )
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src) if not previous else os.pathsep.join((str(src), previous))
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
