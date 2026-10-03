"""Tests for conversation trimming."""

from lupaxa.basic_cli_chatbot.chat import begin_turn, trim


def _pair(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def test_zero_keeps_the_whole_list() -> None:
    messages = _pair("one", "a") + _pair("two", "b")
    trim(messages, 0)
    assert [item["content"] for item in messages] == ["one", "a", "two", "b"]


def test_two_keeps_the_newest_pair() -> None:
    messages = _pair("one", "a") + _pair("two", "b")
    trim(messages, 2)
    assert messages == _pair("two", "b")


def test_odd_limit_keeps_the_new_user_row() -> None:
    messages = _pair("one", "a") + _pair("two", "b")
    begin_turn("three", messages, 3)
    assert messages == [{"role": "user", "content": "three"}]


def test_limit_two_after_a_new_question_drops_older_pairs() -> None:
    messages = _pair("one", "a") + _pair("two", "b")
    begin_turn("three", messages, 2)
    assert messages == [{"role": "user", "content": "three"}]


def test_ask_passes_system_prompt_after_trim() -> None:
    from typing import cast

    from lupaxa.basic_cli_chatbot.chat import SYSTEM_PROMPT, ask
    from lupaxa.basic_cli_chatbot.providers.base import Provider

    class _Provider:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] | None = None
            self.prompt = ""

        def chat(self, messages, model, system_prompt):
            self.messages = messages
            self.prompt = system_prompt
            return "ok"

    conversation = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "b"},
    ]
    provider = _Provider()
    answer = ask(
        "three",
        conversation,
        cast(Provider, provider),
        "m",
        max_context_messages=2,
    )
    assert answer == "ok"
    assert provider.prompt == SYSTEM_PROMPT
    assert provider.messages == [{"role": "user", "content": "three"}]
    assert conversation[-1] == {"role": "assistant", "content": "ok"}
