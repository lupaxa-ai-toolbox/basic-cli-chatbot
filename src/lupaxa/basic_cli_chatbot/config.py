"""YAML profile file. Not part of the public package API."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .providers import KNOWN

_ALLOWED = frozenset(
    {"provider", "model", "ollama-host", "max-context-messages", "stream", "timeout"}
)


class ConfigError(Exception):
    """Config failure. The message is the text after ``Error: ``."""


@dataclass(frozen=True)
class Profile:
    """One named chat setup. Optional fields stay unset when the file omits them."""

    name: str
    provider: str
    model: str
    ollama_host: str | None = None
    max_context_messages: int | None = None
    stream: bool | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class ConfigFile:
    """Profiles in file order."""

    path: Path
    profiles: tuple[Profile, ...]


class _Loader(yaml.SafeLoader):
    """Safe loader that rejects a repeated string key."""


def _construct_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    seen: set[str] = set()
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError:
            raise ConfigError("profile names must be strings") from None
        if isinstance(key, str):
            folded = key.strip().casefold()
            if folded in seen:
                raise ConfigError(f"duplicate profile '{folded}'")
            seen.add(folded)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def default_config_path() -> Path:
    """Return the user config path. ``XDG_CONFIG_HOME`` wins when it is non-empty."""
    raw = os.environ.get("XDG_CONFIG_HOME")
    base = Path(raw.strip()) if raw is not None and raw.strip() else Path.home() / ".config"
    return base / "basic-cli-chatbot" / "config.yaml"


def load_config(path: Path) -> ConfigFile:
    """Read one YAML document. Raise ``ConfigError`` on the first problem."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except OSError:
        raise ConfigError(f"unable to read config file: {path}") from None
    try:
        documents = list(yaml.load_all(text, Loader=_Loader))
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        cause = exc.__cause__
        if isinstance(cause, ConfigError):
            raise cause from None
        detail = str(exc).splitlines()
        first = detail[0] if detail and detail[0].strip() else "invalid YAML"
        raise ConfigError(f"invalid config file: {first}") from None
    if len(documents) > 1:
        raise ConfigError("config file must be a single document")
    root: object = documents[0] if documents else None
    return ConfigFile(path=path, profiles=_profiles(root))


def find_profile(config: ConfigFile, name: str) -> Profile:
    """Return the profile whose stripped name matches, ignoring case."""
    key = name.strip().casefold()
    if not key:
        raise ConfigError("profile is required")
    for profile in config.profiles:
        if profile.name.casefold() == key:
            return profile
    known = ", ".join(profile.name for profile in config.profiles)
    raise ConfigError(f"unknown profile '{key}' ({known})")


def _profiles(root: object) -> tuple[Profile, ...]:
    if root is None:
        root = {}
    if not isinstance(root, dict):
        raise ConfigError("config file must be a mapping")
    profiles: list[Profile] = []
    for raw_name, raw_body in root.items():
        if not isinstance(raw_name, str):
            raise ConfigError("profile names must be strings")
        name = raw_name.strip()
        if not name:
            raise ConfigError("profile name is empty")
        if not isinstance(raw_body, dict):
            raise ConfigError(f"profile '{name}' must be a mapping")
        profiles.append(_profile(name, raw_body))
    return tuple(profiles)


def _profile(name: str, body: dict[object, object]) -> Profile:
    for key in body:
        if not isinstance(key, str) or key not in _ALLOWED:
            shown = key if isinstance(key, str) else str(key)
            raise ConfigError(f"profile '{name}' has unknown key '{shown}'")
    provider = _required_text(name, body, "provider")
    if provider.strip().casefold() not in KNOWN:
        joined = ", ".join(KNOWN)
        shown = provider.strip().casefold()
        raise ConfigError(f"profile '{name}' has unknown provider '{shown}' ({joined})")
    model = _required_text(name, body, "model")
    return Profile(
        name=name,
        provider=provider.strip().casefold(),
        model=model.strip(),
        ollama_host=_optional_host(name, body),
        max_context_messages=_optional_limit(name, body),
        stream=_optional_stream(name, body),
        timeout=_optional_timeout(name, body),
    )


def _required_text(name: str, body: dict[object, object], key: str) -> str:
    if key not in body or body[key] is None:
        raise ConfigError(f"profile '{name}' is missing {key}")
    value = body[key]
    if not isinstance(value, str):
        raise ConfigError(f"profile '{name}' {key} must be a string")
    if not value.strip():
        raise ConfigError(f"profile '{name}' is missing {key}")
    return value


def _optional_host(name: str, body: dict[object, object]) -> str | None:
    if "ollama-host" not in body:
        return None
    value = body["ollama-host"]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"profile '{name}' ollama-host must be a non-empty string")
    return value.strip()


def _optional_limit(name: str, body: dict[object, object]) -> int | None:
    if "max-context-messages" not in body:
        return None
    value = body["max-context-messages"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"profile '{name}' max-context-messages must be an integer")
    if value < 0:
        raise ConfigError(f"profile '{name}' max-context-messages must be 0 or greater")
    return value


def _optional_stream(name: str, body: dict[object, object]) -> bool | None:
    if "stream" not in body:
        return None
    value = body["stream"]
    if not isinstance(value, bool):
        raise ConfigError(f"profile '{name}' stream must be true or false")
    return value


def _optional_timeout(name: str, body: dict[object, object]) -> float | None:
    if "timeout" not in body:
        return None
    value = body["timeout"]
    message = f"profile '{name}' timeout must be a number of seconds greater than 0"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(message)
    try:
        number = float(value)
    except OverflowError:
        raise ConfigError(message) from None
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(message)
    return number
