from typing import Any, Callable

from openai import OpenAI

from seer.automation.agent.types import Message, Usage


class LlmClient:
    def __init__(self):
        self.openai_client = OpenAI()

    def completion(self, model: str, messages: list[Message], **chat_completion_kwargs):
        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            **chat_completion_kwargs,
        )

        response_message = completion.choices[0].message.content

        usage = Usage()
        if completion.usage:
            usage.completion_tokens += completion.usage.completion_tokens
            usage.prompt_tokens += completion.usage.prompt_tokens
            usage.total_tokens += completion.usage.total_tokens

        return response_message, usage

    def completion_with_parser(
        self,
        model: str,
        messages: list[Message],
        parser: Callable[[str], Any],
        **chat_completion_kwargs,
    ):
        response_message, usage = self.completion(model, messages, **chat_completion_kwargs)
        return parser(response_message), usage
