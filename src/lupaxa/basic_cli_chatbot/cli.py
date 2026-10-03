"""Command-line interface for basic-cli-chatbot."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from .chat import (
    SYSTEM_PROMPT,
    ask,
    begin_turn,
    fail_turn,
    finish_turn,
)
from .config import ConfigError, ConfigFile, default_config_path, find_profile, load_config
from .providers import KNOWN, Provider, build
from .providers.catalog import CATALOG, gemini_timeout_ms
from .providers.errors import ChatError
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A simple command-line AI chatbot.")
    parser.add_argument(
        "--provider",
        help="Chat provider: openai, gemini, grok, or ollama.",
    )
    parser.add_argument("--model", help="Model name for the selected provider.")
    parser.add_argument(
        "--config",
        help="Profile file. Defaults to the user config path.",
    )
    parser.add_argument(
        "--profile",
        help="Profile name from the config file.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama server URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--max-context-messages",
        type=int,
        default=None,
        help="Maximum user and assistant rows to keep. 0 means unlimited.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Reply timeout in seconds.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the effective settings and exit.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check the config file and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model names for the selected provider and exit.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the reply as it arrives (default).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print the reply in one piece.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit.",
    )
    parser.add_argument("question", nargs="?", help="Ask one question and exit.")
    return parser


def _flag_or_env(flag: str | None, env_name: str) -> str | None:
    if flag is not None:
        text = flag.strip()
    else:
        raw = os.environ.get(env_name)
        if raw is None:
            return None
        text = raw.strip()
    if not text:
        return None
    return text


def _resolve_provider(flag: str | None) -> str | None:
    text = _flag_or_env(flag, "BASIC_CLI_CHATBOT_PROVIDER")
    if text is None:
        return None
    return text.casefold()


def _resolve_model(flag: str | None) -> str | None:
    return _flag_or_env(flag, "BASIC_CLI_CHATBOT_MODEL")


def _env_timeout() -> float | None:
    raw = os.environ.get("BASIC_CLI_CHATBOT_TIMEOUT")
    if raw is None:
        return None
    text = raw.strip()
    message = "BASIC_CLI_CHATBOT_TIMEOUT must be a number of seconds greater than 0"
    if not text:
        raise ConfigError(message)
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(message) from None
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(message)
    return value


def _resolve_timeout(
    flag: float | None,
    profile_value: float | None,
    provider_name: str | None,
) -> float | None:
    if flag is not None:
        return flag
    if profile_value is not None:
        return profile_value
    chosen = _env_timeout()
    if chosen is not None:
        return chosen
    if provider_name is None or provider_name not in CATALOG:
        return None
    return float(CATALOG[provider_name].timeout_seconds)


def _format_timeout(timeout: float | None) -> str:
    if timeout is None:
        return "not set"
    if timeout.is_integer():
        return str(int(timeout))
    text = format(timeout, "f").rstrip("0").rstrip(".")
    return text


def _reject_gemini_timeout(provider_name: str, timeout: float) -> bool:
    if provider_name != "gemini":
        return False
    try:
        gemini_timeout_ms(timeout)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return True
    return False


def format_show_config(
    provider: str | None,
    model: str | None,
    max_context_messages: int,
    ollama_host: str,
    timeout: float | None,
    profile: str | None = None,
    config_path: Path | None = None,
) -> None:
    """Print the effective settings. Never print a secret."""
    facts = CATALOG.get(provider) if provider is not None else None
    lines = [
        f"provider: {provider or 'not set'}",
        f"model: {model or 'not set'}",
        f"mode: {facts.mode if facts is not None else 'not set'}",
        f"max-context-messages: {max_context_messages}",
        f"timeout: {_format_timeout(timeout)}",
    ]
    if provider == "ollama":
        lines.append(f"ollama-host: {ollama_host}")
    if facts is not None and facts.api_key_env is not None:
        state = "configured" if os.environ.get(facts.api_key_env) else "not configured"
        lines.append(f"api-key: {state}")
    if profile is not None and config_path is not None:
        lines.append(f"profile: {profile}")
        lines.append(f"config: {config_path}")
    print("\n".join(lines))


_STREAM_NOTICE = "Streaming is unavailable; using a single response."


def _print_stream_notice(*, interactive: bool) -> None:
    print(_STREAM_NOTICE, file=sys.stdout if interactive else sys.stderr)


def _print_reply(answer: str, *, interactive: bool) -> None:
    if interactive:
        print(f"\nAI: {answer}\n")
    else:
        print(answer)


def _close_stream(text: str, *, interactive: bool) -> None:
    if not text.endswith("\n"):
        print(flush=True)
    if interactive:
        print(flush=True)


def _collect_stream(
    conversation: list[dict[str, str]],
    provider: Provider,
    model: str,
    *,
    interactive: bool,
) -> tuple[str, bool]:
    """Return the reply and whether its text was already written."""
    if not provider.supports_stream:
        _print_stream_notice(interactive=interactive)
        return provider.chat(list(conversation), model, SYSTEM_PROMPT), False

    parts: list[str] = []
    try:
        for delta in provider.stream(list(conversation), model, SYSTEM_PROMPT):
            if interactive and not parts:
                print("\nAI: ", end="", flush=True)
            print(delta, end="", flush=True)
            parts.append(delta)
    except Exception:
        if parts:
            raise
        _print_stream_notice(interactive=interactive)
        return provider.chat(list(conversation), model, SYSTEM_PROMPT), False

    if not parts:
        raise ChatError("the provider returned an invalid response")
    text = "".join(parts)
    _close_stream(text, interactive=interactive)
    return text, True


def _stream_reply(
    question: str,
    conversation: list[dict[str, str]],
    provider: Provider,
    model: str,
    max_context_messages: int,
    *,
    interactive: bool,
) -> None:
    begin_turn(question, conversation, max_context_messages)
    try:
        answer, streamed = _collect_stream(
            conversation,
            provider,
            model,
            interactive=interactive,
        )
    except Exception:
        fail_turn(conversation)
        raise
    finish_turn(conversation, answer)
    if not streamed:
        _print_reply(answer, interactive=interactive)


def run_oneshot(
    question: str,
    provider_name: str,
    model: str,
    ollama_host: str,
    max_context_messages: int,
    *,
    stream: bool,
    timeout: float,
) -> int:
    stripped = question.strip()
    if not stripped:
        print("Error: question is empty", file=sys.stderr)
        return 2
    try:
        provider = build(provider_name, ollama_host=ollama_host, timeout=timeout)
        if stream:
            _stream_reply(
                stripped,
                [],
                provider,
                model,
                max_context_messages,
                interactive=False,
            )
        else:
            answer = ask(
                stripped,
                [],
                provider,
                model,
                max_context_messages=max_context_messages,
            )
            print(answer)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def run_list_models(provider_name: str, ollama_host: str, *, timeout: float) -> int:
    try:
        provider = build(provider_name, ollama_host=ollama_host, timeout=timeout)
        names = provider.list_models()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for name in names:
        print(name)
    return 0


def _print_slash(
    command: str,
    *,
    provider_name: str,
    model: str,
    conversation: list[dict[str, str]],
    max_context_messages: int,
) -> bool:
    if command == "/context":
        limit = "unlimited" if max_context_messages == 0 else str(max_context_messages)
        print(f"context: {len(conversation)} messages, limit {limit}")
        return True
    if command == "/provider":
        print(f"provider: {provider_name}")
        return True
    if command == "/model":
        print(f"model: {model}")
        return True
    return False


def run_interactive(
    provider_name: str,
    model: str,
    ollama_host: str,
    max_context_messages: int,
    *,
    stream: bool,
    timeout: float,
) -> int:
    mode = CATALOG[provider_name].mode
    print("Simple AI Chatbot")
    print(f"Provider: {provider_name}")
    print(f"Model: {model}")
    print(f"Mode: {mode}")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'clear' to clear the conversation context.\n")
    conversation: list[dict[str, str]] = []
    provider: Provider | None = None
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if not question:
            continue
        command = question.casefold()
        if command in {"exit", "quit"}:
            print("Goodbye!")
            return 0
        if command == "clear":
            conversation.clear()
            print("\nConversation context cleared.\n")
            continue
        if _print_slash(
            command,
            provider_name=provider_name,
            model=model,
            conversation=conversation,
            max_context_messages=max_context_messages,
        ):
            continue
        if provider is None:
            try:
                provider = build(provider_name, ollama_host=ollama_host, timeout=timeout)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return 0
            except Exception as exc:
                print(f"\nError: {exc}\n")
                continue
        try:
            if stream:
                _stream_reply(
                    question,
                    conversation,
                    provider,
                    model,
                    max_context_messages,
                    interactive=True,
                )
            else:
                answer = ask(
                    question,
                    conversation,
                    provider,
                    model,
                    max_context_messages=max_context_messages,
                )
                print(f"\nAI: {answer}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except Exception as exc:
            print(f"\nError: {exc}\n")
            continue


_HOST = "http://localhost:11434"


def _config_path(flag: str | None) -> tuple[Path, bool]:
    if flag is not None:
        return Path(flag).expanduser(), True
    return default_config_path(), False


def _load_for_default(path: Path, explicit: bool) -> ConfigFile | None:
    try:
        return load_config(path)
    except ConfigError as exc:
        if not explicit and str(exc) == f"config file not found: {path}":
            return None
        raise


def _optional_host(flag: str | None, profile_value: str | None) -> str:
    if flag is not None:
        return flag
    if profile_value is not None:
        return profile_value
    return _HOST


def _optional_limit(flag: int | None, profile_value: int | None) -> int:
    if flag is not None:
        return flag
    if profile_value is not None:
        return profile_value
    return 0


def _stream_choice(stream: bool, no_stream: bool, profile_stream: bool | None) -> bool:
    if stream:
        return True
    if no_stream:
        return False
    if profile_stream is not None:
        return profile_stream
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"basic-cli-chatbot {__version__}")
        return 0
    try:
        return _run(args)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    choices = (args.question is not None, args.list_models, args.show_config, args.validate)
    if sum(choices) > 1:
        print(
            "Error: choose one of a question, --list-models, --show-config, or --validate",
            file=sys.stderr,
        )
        return 2
    if (args.stream or args.no_stream) and (args.list_models or args.show_config or args.validate):
        print("Error: --stream applies only to a chat", file=sys.stderr)
        return 2
    if args.stream and args.no_stream:
        print("Error: --stream cannot be combined with --no-stream", file=sys.stderr)
        return 2
    if args.max_context_messages is not None and args.max_context_messages < 0:
        print("Error: --max-context-messages must be 0 or greater", file=sys.stderr)
        return 2
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        print(
            "Error: --timeout must be a number of seconds greater than 0",
            file=sys.stderr,
        )
        return 2
    if args.validate and (
        args.profile is not None or args.provider is not None or args.model is not None
    ):
        print(
            "Error: --validate cannot be combined with --profile, --provider, or --model",
            file=sys.stderr,
        )
        return 2
    if args.profile is not None and (args.provider is not None or args.model is not None):
        print("Error: --profile cannot be combined with --provider or --model", file=sys.stderr)
        return 2
    if args.profile is not None and not args.profile.strip():
        print("Error: profile is required", file=sys.stderr)
        return 2

    path, explicit = _config_path(args.config)
    profile = None
    config_path: Path | None = None
    provider_name: str | None
    model: str | None
    if args.validate:
        loaded = load_config(path)
        names = ", ".join(item.name for item in loaded.profiles)
        second = f"profiles: {names}" if names else "profiles:"
        print(f"config: ok\n{second}")
        return 0
    if args.profile is not None:
        loaded = load_config(path)
        profile = find_profile(loaded, args.profile)
        config_path = path
        provider_name = profile.provider
        model = profile.model
    elif args.provider is None and args.model is None:
        default_file = _load_for_default(path, explicit)
        if default_file is not None:
            matches = [item for item in default_file.profiles if item.name.casefold() == "default"]
            if matches:
                profile = matches[0]
                config_path = path
                provider_name = profile.provider
                model = profile.model
            else:
                provider_name = _resolve_provider(None)
                model = _resolve_model(None)
        else:
            provider_name = _resolve_provider(None)
            model = _resolve_model(None)
    else:
        provider_name = _resolve_provider(args.provider)
        model = _resolve_model(args.model)

    if profile is None:
        ollama_host = args.ollama_host if args.ollama_host is not None else _HOST
        max_context = 0 if args.max_context_messages is None else args.max_context_messages
        stream = _stream_choice(args.stream, args.no_stream, None)
    else:
        ollama_host = _optional_host(args.ollama_host, profile.ollama_host)
        max_context = _optional_limit(args.max_context_messages, profile.max_context_messages)
        stream = _stream_choice(args.stream, args.no_stream, profile.stream)

    profile_timeout = None if profile is None else profile.timeout
    timeout = _resolve_timeout(args.timeout, profile_timeout, provider_name)

    if args.show_config:
        format_show_config(
            provider_name,
            model,
            max_context,
            ollama_host,
            timeout,
            profile=None if profile is None else profile.name,
            config_path=config_path,
        )
        return 0
    names = ", ".join(KNOWN)
    if provider_name is None:
        print(f"Error: provider is required ({names})", file=sys.stderr)
        return 2
    if provider_name not in KNOWN:
        print(f"Error: unknown provider '{provider_name}' ({names})", file=sys.stderr)
        return 2
    if args.list_models:
        if timeout is None or _reject_gemini_timeout(provider_name, timeout):
            return 2
        return run_list_models(
            provider_name,
            ollama_host,
            timeout=timeout,
        )
    if model is None:
        print("Error: model is required", file=sys.stderr)
        return 2
    if timeout is None or _reject_gemini_timeout(provider_name, timeout):
        return 2
    if args.question is None:
        return run_interactive(
            provider_name,
            model,
            ollama_host,
            max_context,
            stream=stream,
            timeout=timeout,
        )
    return run_oneshot(
        args.question,
        provider_name,
        model,
        ollama_host,
        max_context,
        stream=stream,
        timeout=timeout,
    )
