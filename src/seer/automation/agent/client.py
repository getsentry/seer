import dataclasses
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar

from langfuse.openai import openai
from openai.types.chat import ChatCompletion

from seer.automation.agent.models import Message, Usage
from seer.bootup import module, stub_module

T = TypeVar("T")


class LlmClient(ABC):
    @abstractmethod
    def completion(
        self, messages: list[Message], model: str, **chat_completion_kwargs
    ) -> tuple[Message, Usage]:
        pass

    def completion_with_parser(
        self,
        messages: list[Message],
        parser: Callable[[str | None], T],
        model: str,
        **chat_completion_kwargs,
    ) -> tuple[T, Message, Usage]:
        message, usage = self.completion(messages, model, **chat_completion_kwargs)

        return parser(message.content), message, usage


DEFAULT_GPT_MODEL = "gpt-4o-2024-05-13"


class GptClient(LlmClient):
    def __init__(self):
        self.openai_client = openai.Client()

    def completion(
        self, messages: list[Message], model=DEFAULT_GPT_MODEL, **chat_completion_kwargs
    ):
        completion: ChatCompletion = self.openai_client.chat.completions.create(
            model=model,
            messages=[message.to_openai_message() for message in messages],
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

    def completion_with_parser(
        self,
        messages: list[Message],
        parser: Callable[[str | None], T],
        model=DEFAULT_GPT_MODEL,
        **chat_completion_kwargs,
    ) -> tuple[T, Message, Usage]:
        message, usage = self.completion(messages, model, **chat_completion_kwargs)

        return parser(message.content), message, usage

    def json_completion(
        self, messages: list[Message], model=DEFAULT_GPT_MODEL, **chat_completion_kwargs
    ) -> tuple[dict[str, Any] | None, Message, Usage]:
        return self.completion_with_parser(
            messages,
            parser=lambda x: json.loads(x) if x else None,
            model=model,
            response_format={"type": "json_object"},
            **chat_completion_kwargs,
        )


@module.provider
def provide_gpt_client() -> GptClient:
    return GptClient()


GptCompletionHandler = Callable[[list[Message], dict[str, Any]], Optional[tuple[Message, Usage]]]


@dataclasses.dataclass
class DummyGptClient(GptClient):
    handlers: list[GptCompletionHandler] = dataclasses.field(default_factory=list)
    missed_calls: list[tuple[list[Message], dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )

    def completion(self, messages: list[Message], model="test-gpt", **chat_completion_kwargs):
        for handler in self.handlers:
            result = handler(messages, chat_completion_kwargs)
            if result:
                return result
        self.missed_calls.append((messages, chat_completion_kwargs))
        return Message(), Usage()


@stub_module.provider
def provide_stub_gpt_client() -> GptClient:
    return DummyGptClient()
