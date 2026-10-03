"""Ollama provider. Not part of the public package API."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager

import httpx
from ollama import Client, ResponseError

from .catalog import CATALOG
from .errors import ChatError, one_line


class OllamaProvider:
    """Chat through the official Ollama client."""

    mode = CATALOG["ollama"].mode
    supports_stream = True

    def __init__(self, host: str, timeout: float) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> str:
        payload = _messages(messages, system_prompt)
        with _session(self, model) as client:
            try:
                response = client.chat(model=model, messages=payload, stream=False)
                content = response.message.content
            except AttributeError as exc:
                raise ChatError("the provider returned an invalid response") from exc
            if content is None or not str(content).strip():
                raise ChatError("the provider returned an invalid response")
            return str(content)

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> Iterator[str]:
        payload = _messages(messages, system_prompt)
        with _session(self, model) as client:
            for chunk in client.chat(model=model, messages=payload, stream=True):
                try:
                    content = chunk.message.content
                except AttributeError as exc:
                    raise ChatError("the provider returned an invalid response") from exc
                if content:
                    yield str(content)

    def list_models(self) -> list[str]:
        with _session(self) as client:
            try:
                listed = client.list()
                names = [item.model for item in listed.models]
            except AttributeError as exc:
                raise ChatError("the provider returned an invalid response") from exc
            if any(name is None or not str(name).strip() for name in names):
                raise ChatError("the provider returned an invalid response")
            return [str(name) for name in names]


def _timeout(seconds: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=min(5.0, seconds),
        read=seconds,
        write=min(30.0, seconds),
        pool=min(5.0, seconds),
    )


def _messages(
    messages: list[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, *list(messages)]


def _missing_model(model: str) -> ChatError:
    return ChatError(f"Ollama model '{model}' is not installed.\nollama pull {model}")


@contextmanager
def _session(provider: OllamaProvider, model: str | None = None) -> Generator[Client, None, None]:
    client = Client(host=provider.host, timeout=_timeout(provider.timeout))
    try:
        yield client
    except ChatError:
        raise
    except httpx.TimeoutException:
        raise ChatError("Ollama request timed out") from None
    except ConnectionError:
        raise ChatError(f"unable to connect to Ollama at {provider.host}") from None
    except ResponseError as exc:
        if model is not None and exc.status_code == 404:
            raise _missing_model(model) from None
        raise ChatError(one_line(exc)) from None
    except Exception as exc:
        raise ChatError(one_line(exc)) from None
    finally:
        client.close()
