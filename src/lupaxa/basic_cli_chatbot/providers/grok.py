"""Grok provider. Not part of the public package API."""

from __future__ import annotations

import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager

import grpc
from xai_sdk import Client
from xai_sdk.chat import assistant, system, user
from xai_sdk.proto import chat_pb2

from .catalog import CATALOG
from .errors import ChatError, looks_like_context_limit, one_line


class GrokProvider:
    """Chat through the official xAI SDK."""

    mode = CATALOG["grok"].mode
    supports_stream = True

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> str:
        payload = _messages(messages, system_prompt)
        with _session(self.timeout, model) as client:
            chat = client.chat.create(
                model=model,
                messages=payload,
                store_messages=False,
            )
            response = chat.sample()
            try:
                content = response.content
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
        with _session(self.timeout, model) as client:
            chat = client.chat.create(
                model=model,
                messages=payload,
                store_messages=False,
            )
            for _response, chunk in chat.stream():
                try:
                    content = chunk.content
                except AttributeError as exc:
                    raise ChatError("the provider returned an invalid response") from exc
                if content:
                    yield str(content)

    def list_models(self) -> list[str]:
        with _session(self.timeout, "") as client:
            names: list[str] = []
            for model in client.models.list_language_models():
                name = getattr(model, "name", None)
                if name is None or not str(name).strip():
                    raise ChatError("the provider returned an invalid response")
                names.append(str(name))
            return names


def _messages(
    messages: list[dict[str, str]],
    system_prompt: str,
) -> list[chat_pb2.Message]:
    payload: list[chat_pb2.Message] = [system(system_prompt)]
    for message in messages:
        content = message["content"]
        role = message["role"]
        if role == "user":
            payload.append(user(content))
        elif role == "assistant":
            payload.append(assistant(content))
        else:
            raise ChatError("the provider returned an invalid response")
    return payload


def _client(timeout: float) -> Client:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise ChatError("XAI_API_KEY is not configured")
    return Client(
        api_key=key,
        timeout=timeout,
        channel_options=[("grpc.enable_retries", 0)],
    )


@contextmanager
def _session(timeout: float, model: str) -> Generator[Client, None, None]:
    client = _client(timeout)
    try:
        yield client
    except ChatError:
        raise
    except grpc.RpcError as exc:
        raise _translate(exc, model) from None
    except OSError:
        raise ChatError("network error talking to Grok") from None
    except Exception as exc:
        raise ChatError(one_line(exc)) from None
    finally:
        client.close()


def _translate(exc: grpc.RpcError, model: str) -> ChatError:
    code = exc.code()
    if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
        return ChatError("Grok authentication failed")
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return ChatError("Grok rate limit reached")
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return ChatError("Grok request timed out")
    if code == grpc.StatusCode.NOT_FOUND:
        return ChatError(f"model '{model}' is not available")
    if code == grpc.StatusCode.INVALID_ARGUMENT and looks_like_context_limit(str(exc)):
        return ChatError("the conversation is too long for this model")
    if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL):
        return ChatError("Grok is unavailable")
    return ChatError(one_line(exc))
