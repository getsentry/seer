import dataclasses
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar

import openai_multi_tool_use_parallel_patch  # noqa - import applies the patch
from openai import OpenAI
from openai.types.chat import ChatCompletion

from seer.automation.agent.models import Message, Usage

T = TypeVar("T")


class LlmClient(ABC):
    @abstractmethod
    def completion(
        self, messages: list[Message], **chat_completion_kwargs
    ) -> tuple[Message, Usage]:
        pass

    def completion_with_parser(
        self,
        messages: list[Message],
        parser: Callable[[str | None], T],
        **chat_completion_kwargs,
    ) -> tuple[T, Message, Usage]:
        message, usage = self.completion(messages, **chat_completion_kwargs)

        return parser(message.content), message, usage


class GptClient(LlmClient):
    def __init__(self, model: str = "gpt-4o-2024-05-13"):
        self.model = model
        self.openai_client = OpenAI()

    def completion(self, messages: list[Message], **chat_completion_kwargs):
        completion: ChatCompletion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=0.0,
            **chat_completion_kwargs,
        )

        message = Message(**completion.choices[0].message.dict())

        usage = Usage()
        if completion.usage:
            usage.completion_tokens += completion.usage.completion_tokens
            usage.prompt_tokens += completion.usage.prompt_tokens
            usage.total_tokens += completion.usage.total_tokens

        return message, usage

    def json_completion(
        self, messages: list[Message], **chat_completion_kwargs
    ) -> tuple[dict[str, Any] | None, Message, Usage]:
        return self.completion_with_parser(
            messages,
            parser=lambda x: json.loads(x) if x else None,
            response_format={"type": "json_object"},
            **chat_completion_kwargs,
        )


GptCompletionHandler = Callable[[list[Message], dict[str, Any]], Optional[tuple[Message, Usage]]]


@dataclasses.dataclass
class DummyGptClient(GptClient):
    handlers: list[GptCompletionHandler] = dataclasses.field(default_factory=list)
    missed_calls: list[tuple[list[Message], dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )

    def completion(self, messages: list[Message], **chat_completion_kwargs):
        for handler in self.handlers:
            result = handler(messages, chat_completion_kwargs)
            if result:
                return result
        self.missed_calls.append((messages, chat_completion_kwargs))
        return Message(), Usage()
