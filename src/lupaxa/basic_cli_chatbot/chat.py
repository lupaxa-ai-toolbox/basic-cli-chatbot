"""Conversation history helper. Not part of the public package API."""

from __future__ import annotations

from .providers.base import Provider

SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. Answer questions clearly and concisely."
)


def trim(conversation: list[dict[str, str]], max_context_messages: int) -> None:
    """Drop oldest complete user/assistant pairs until the list fits.

    ``0`` means unlimited. An odd cap uses the next lower even number.
    A trailing user row with no assistant reply is kept.
    """
    if max_context_messages == 0:
        return
    limit = max_context_messages - (max_context_messages % 2)
    while len(conversation) > limit:
        complete_pair = (
            len(conversation) >= 2
            and conversation[0].get("role") == "user"
            and conversation[1].get("role") == "assistant"
        )
        if not complete_pair:
            break
        if conversation[-1].get("role") == "user" and len(conversation) < 3:
            break
        del conversation[:2]


def begin_turn(
    question: str,
    conversation: list[dict[str, str]],
    max_context_messages: int,
) -> None:
    """Append the user row, then trim."""
    conversation.append({"role": "user", "content": question})
    trim(conversation, max_context_messages)


def finish_turn(conversation: list[dict[str, str]], answer: str) -> None:
    """Append the assistant row."""
    conversation.append({"role": "assistant", "content": answer})


def fail_turn(conversation: list[dict[str, str]]) -> None:
    """Remove the trailing user row after a failed request."""
    if conversation and conversation[-1].get("role") == "user":
        conversation.pop()


def ask(
    question: str,
    conversation: list[dict[str, str]],
    provider: Provider,
    model: str,
    *,
    max_context_messages: int,
) -> str:
    """Append one user turn, request a reply, and append that reply."""
    begin_turn(question, conversation, max_context_messages)
    try:
        answer = provider.chat(list(conversation), model, SYSTEM_PROMPT)
    except Exception:
        fail_turn(conversation)
        raise
    finish_turn(conversation, answer)
    return answer
