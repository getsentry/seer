import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar

import openai_multi_tool_use_parallel_patch  # import applies the patch
from langsmith import wrappers
from openai import OpenAI
from openai.types.chat import ChatCompletion

from seer.automation.agent.models import Message, Usage

T = TypeVar("T")


class LlmClient(ABC):
    @abstractmethod
    def completion(
        self, model: str, messages: list[Message], **chat_completion_kwargs
    ) -> tuple[Message, Usage]:
        pass

    def completion_with_parser(
        self,
        model: str,
        messages: list[Message],
        parser: Callable[[str | None], T],
        **chat_completion_kwargs,
    ) -> tuple[T, Message, Usage]:
        message, usage = self.completion(model, messages, **chat_completion_kwargs)

        return parser(message.content), message, usage


class GptClient(LlmClient):
    def __init__(self):
        self.openai_client = wrappers.wrap_openai(OpenAI())

    def completion(self, model: str, messages: list[Message], **chat_completion_kwargs):
        completion: ChatCompletion = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            **chat_completion_kwargs,
        )

        message = Message(**completion.choices[0].message.dict())

        usage = Usage()
        if completion.usage:
            usage.completion_tokens += completion.usage.completion_tokens
            usage.prompt_tokens += completion.usage.prompt_tokens
            usage.total_tokens += completion.usage.total_tokens

        return message, usage


GptCompletionHandler = Callable[
    [str, list[Message], dict[str, Any]], Optional[tuple[Message, Usage]]
]


@dataclasses.dataclass
class DummyGptClient(GptClient):
    handlers: list[GptCompletionHandler] = dataclasses.field(default_factory=list)
    missed_calls: list[tuple[str, list[Message], dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )

    def completion(self, model: str, messages: list[Message], **chat_completion_kwargs):
        for handler in self.handlers:
            result = handler(model, messages, chat_completion_kwargs)
            if result:
                return result
        self.missed_calls.append((model, messages, chat_completion_kwargs))
        return Message(), Usage()
