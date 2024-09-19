import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar

import anthropic
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai

from seer.automation.agent.models import Message, ToolCall, Usage
from seer.automation.agent.tools import FunctionTool
from seer.automation.agent.utils import extract_json_from_text
from seer.bootup import module, stub_module
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LlmClient(ABC):
    @abstractmethod
    def completion(
        self,
        messages: list[Message],
        model: str | None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = [],
        response_format: Optional[dict] = None,
    ) -> tuple[Message, Usage]:
        raise NotImplementedError

    @abstractmethod
    def completion_with_parser(
        self,
        messages: list[Message],
        parser: Callable[[str | None], T],
        model: str,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = None,
        response_format: Optional[dict] = None,
    ) -> tuple[T, Message, Usage]:
        message, usage = self.completion(
            messages,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            response_format=response_format,
        )
        return parser(message.content), message, usage

    @abstractmethod
    def json_completion(
        self, messages: list[Message], model: str, system_prompt: Optional[str] = None
    ) -> tuple[dict[str, Any] | None, Message, Usage]:
        return self.completion_with_parser(
            messages,
            model=model,
            system_prompt=system_prompt,
            parser=lambda x: extract_json_from_text(x),
            response_format={"type": "json_object"},
        )


DEFAULT_GPT_MODEL = "gpt-4o-2024-05-13"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet@20240620"


class GptClient(LlmClient):
    def __init__(self):
        self.openai_client = openai.Client()

    def completion(
        self,
        messages: list[Message],
        model: str | None = DEFAULT_GPT_MODEL,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = None,
        response_format: Optional[dict] = None,
    ):
        message_dicts = [message.to_message() for message in messages]
        if system_prompt:
            message_dicts.insert(0, Message(role="system", content=system_prompt).to_message())

        tool_dicts = (
            [tool.to_dict(model="gpt") for tool in tools]
            if tools and len(tools) > 0
            else openai.NotGiven()
        )

        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=message_dicts,
            temperature=0.0,
            tools=tool_dicts,
            response_format=response_format if response_format else openai.NotGiven(),
        )

        response = completion.choices[0].message
        tool_calls = (
            [
                ToolCall(id=call.id, function=call.function.name, args=call.function.arguments)
                for call in response.tool_calls
            ]
            if response.tool_calls
            else None
        )
        message = Message(content=response.content, role=response.role, tool_calls=tool_calls)

        usage = Usage(
            completion_tokens=completion.usage.completion_tokens,
            prompt_tokens=completion.usage.prompt_tokens,
            total_tokens=completion.usage.total_tokens,
        )

        return message, usage

    def completion_with_parser(
        self,
        messages: list[Message],
        parser: Callable[[str | None], T],
        model: str = DEFAULT_GPT_MODEL,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        response_format: dict | None = None,
    ) -> tuple[T, Message, Usage]:
        return super().completion_with_parser(
            messages, parser, model, system_prompt, tools, response_format
        )

    def json_completion(
        self,
        messages: list[Message],
        model: str = DEFAULT_GPT_MODEL,
        system_prompt: str | None = None,
    ) -> tuple[dict[str, Any] | None, Message, Usage]:
        return super().json_completion(messages, model, system_prompt)

    def clean_tool_call_assistant_messages(self, messages: list[Message]) -> list[Message]:
        new_messages = []
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                new_messages.append(
                    Message(role="assistant", content=message.content, tool_calls=[])
                )
            elif message.role == "tool":
                new_messages.append(Message(role="user", content=message.content, tool_calls=[]))
            else:
                new_messages.append(message)
        return new_messages


class ClaudeClient(LlmClient):
    @inject
    def __init__(self, config: AppConfig = injected):
        self.anthropic_client = anthropic.AnthropicVertex(
            project_id=config.GOOGLE_CLOUD_PROJECT,
            region="europe-west1" if config.USE_EU_REGION else "us-east5",
        )

    @observe(as_type="generation", name="Claude-generation")
    def completion(
        self,
        messages: list[Message],
        model: str | None = DEFAULT_CLAUDE_MODEL,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = None,
        response_format: Optional[dict] = None,
    ) -> tuple[Message, Usage]:
        if response_format:
            # Claude claims to be reliable at providing structured outputs (like JSON) when prompted,
            # but it does not guarantee it like ChatGPT with a special repsonse_format field
            logger.warning(
                "A response format was specified for this completion, but Claude doesn't guarantee a correctly-formatted output"
            )

        claude_messages = self._format_messages_for_claude_input(messages)
        # ask Claude for a response
        params: dict[str, Any] = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 4096,
            "messages": claude_messages,
        }
        if system_prompt:
            params["system"] = system_prompt
        if tools and len(tools) > 0:
            tool_dicts = [tool.to_dict(model="claude") for tool in tools]
            params["tools"] = tool_dicts
        completion = self.anthropic_client.messages.create(**params)
        message = self._format_claude_response_to_message(completion)

        usage = Usage(
            completion_tokens=completion.usage.output_tokens,
            prompt_tokens=completion.usage.input_tokens,
            total_tokens=completion.usage.input_tokens + completion.usage.output_tokens,
        )
        langfuse_context.update_current_observation(model=model, usage=usage)

        return message, usage

    def completion_with_parser(
        self,
        messages: list[Message],
        parser: Callable[[str | None], T],
        model: str = DEFAULT_CLAUDE_MODEL,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        response_format: dict | None = None,
    ) -> tuple[T, Message, Usage]:
        return super().completion_with_parser(
            messages, parser, model, system_prompt, tools, response_format
        )

    def json_completion(
        self,
        messages: list[Message],
        model: str = DEFAULT_CLAUDE_MODEL,
        system_prompt: str | None = None,
    ) -> tuple[dict[str, Any] | None, Message, Usage]:
        return super().json_completion(messages, model, system_prompt)

    def _format_messages_for_claude_input(self, messages: list[Message]) -> list[dict]:
        claude_messages = []
        for message in messages:
            if message.role == "tool":  # we're responding to Claude with a tool use result
                claude_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": message.content,
                                "tool_use_id": message.tool_call_id,
                            }
                        ],
                    }
                )
            elif message.role == "tool_use":
                if not message.tool_calls:
                    continue
                for tool_call in message.tool_calls:
                    claude_messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": tool_call.function,
                                    "input": json.loads(tool_call.args),
                                }
                            ],
                        }
                    )
            else:  # normal text message
                claude_messages.append(
                    {"role": message.role, "content": [{"type": "text", "text": message.content}]}
                )
        return claude_messages

    def _format_claude_response_to_message(self, completion: anthropic.types.Message) -> Message:
        message = Message(role=completion.role)
        for block in completion.content:
            if block.type == "text":
                message.content = (
                    block.text
                )  # we're assuming there's only one text block per message
            elif block.type == "tool_use":
                if not message.tool_calls:
                    message.tool_calls = []
                message.tool_calls.append(
                    ToolCall(id=block.id, function=block.name, args=json.dumps(block.input))
                )
                message.role = "tool_use"
                message.tool_call_id = message.tool_calls[
                    0
                ].id  # assumes we get only 1 tool call at a time, but we really don't use this field for tool_use blocks
        return message


@module.provider
def provide_gpt_client() -> GptClient:
    return GptClient()


@module.provider
def provide_claude_client() -> ClaudeClient:
    return ClaudeClient()


LlmCompletionHandler = Callable[[list[Message], dict[str, Any]], Optional[tuple[Message, Usage]]]


@dataclasses.dataclass
class DummyGptClient(GptClient):
    handlers: list[LlmCompletionHandler] = dataclasses.field(default_factory=list)
    missed_calls: list[tuple[list[Message], dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )

    def completion(
        self,
        messages: list[Message],
        model="test-gpt",
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = [],
        response_format: Optional[dict] = None,
    ):
        for handler in self.handlers:
            result = handler(
                messages,
                {
                    "system_prompt": system_prompt,
                    "tools": tools,
                    "response_format": response_format,
                },
            )
            if result:
                return result
        self.missed_calls.append(
            (
                messages,
                {
                    "system_prompt": system_prompt,
                    "tools": tools,
                    "response_format": response_format,
                },
            )
        )
        return Message(), Usage()


@stub_module.provider
def provide_stub_gpt_client() -> GptClient:
    return DummyGptClient()
