"""OpenAI Responses provider. Not part of the public package API."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.responses import ResponseInputParam

from .catalog import CATALOG
from .errors import (
    ChatError,
    looks_like_context_limit,
    one_line,
)


class OpenAIProvider:
    """Chat through the OpenAI Responses API."""

    mode = CATALOG["openai"].mode
    supports_stream = True

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> str:
        client = _client(self.timeout)
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=cast(ResponseInputParam, list(messages)),
                timeout=self.timeout,
            )
        except ChatError:
            raise
        except OpenAIError as exc:
            raise _translate(exc, model) from None
        except Exception as exc:
            raise ChatError(one_line(exc)) from None
        text = (response.output_text or "").strip()
        if not text:
            raise ChatError("the provider returned an invalid response")
        return response.output_text

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_prompt: str,
    ) -> Iterator[str]:
        client = _client(self.timeout)
        try:
            with client.responses.stream(
                model=model,
                instructions=system_prompt,
                input=cast(ResponseInputParam, list(messages)),
                timeout=self.timeout,
            ) as stream:
                for event in stream:
                    if getattr(event, "type", "") != "response.output_text.delta":
                        continue
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield delta
        except ChatError:
            raise
        except OpenAIError as exc:
            raise _translate(exc, model) from None
        except Exception as exc:
            raise ChatError(one_line(exc)) from None

    def list_models(self) -> list[str]:
        client = _client(self.timeout)
        try:
            return [item.id for item in client.models.list(timeout=self.timeout)]
        except ChatError:
            raise
        except OpenAIError as exc:
            raise _translate(exc, "") from None
        except Exception as exc:
            raise ChatError(one_line(exc)) from None


def _client(timeout: float) -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ChatError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=key, timeout=timeout)


def _translate(exc: OpenAIError, model: str) -> ChatError:
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return ChatError("OpenAI authentication failed")
    if isinstance(exc, RateLimitError):
        return ChatError("OpenAI rate limit reached")
    if isinstance(exc, APITimeoutError):
        return ChatError("OpenAI request timed out")
    if isinstance(exc, APIConnectionError):
        return ChatError("network error talking to OpenAI")
    if isinstance(exc, NotFoundError):
        return ChatError(f"model '{model}' is not available")
    if isinstance(exc, InternalServerError):
        return ChatError("OpenAI is unavailable")
    if isinstance(exc, BadRequestError) and looks_like_context_limit(str(exc)):
        return ChatError("the conversation is too long for this model")
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return ChatError("OpenAI is unavailable")
    return ChatError(one_line(exc))
