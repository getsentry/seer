import json
import logging
import re
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, Type, TypeVar, Union, cast

import anthropic
from anthropic import NOT_GIVEN
from anthropic.types import MessageParam, ToolParam
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel

from seer.automation.agent.models import LlmProviderType, Message, ToolCall, Usage
from seer.automation.agent.tools import FunctionTool
from seer.bootup import module
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class LlmResponseMetadata(BaseModel):
    model: str
    provider_name: LlmProviderType
    usage: Usage


class LlmGenerateTextResponse(BaseModel):
    message: Message
    metadata: LlmResponseMetadata


StructuredOutputType = TypeVar("StructuredOutputType")


class LlmGenerateStructuredResponse(BaseModel, Generic[StructuredOutputType]):
    parsed: StructuredOutputType
    metadata: LlmResponseMetadata


class LlmProviderDefinition(BaseModel):
    model_name: str
    provider_name: LlmProviderType


@dataclass
class OpenAiProvider:
    provider: LlmProviderDefinition

    default_configs = [
        {
            "match": r"^o1-mini.*",
            "temperature": 1.0,
        },
        {
            "match": r"^o1-preview.*",
            "temperature": 1.0,
        },
        {
            "match": r".*",
            "temperature": 0.0,
        },
    ]

    @staticmethod
    def get_client() -> openai.Client:
        return openai.Client()

    @classmethod
    def model(cls, model_name: str) -> "OpenAiProvider":
        return cls(
            provider=LlmProviderDefinition(
                model_name=model_name, provider_name=LlmProviderType.OPENAI
            )
        )

    def _get_config(self, model_name: str):
        for config in self.default_configs:
            if re.match(config["match"], model_name):
                return config
        return None

    def generate_text(
        self,
        *,
        message_dicts: Iterable[ChatCompletionMessageParam],
        tool_dicts: Iterable[ChatCompletionToolParam] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        openai_client = self.get_client()

        config = self._get_config(self.provider.model_name)
        if config:
            if temperature is None:
                temperature = config["temperature"]

        completion = openai_client.chat.completions.create(
            model=self.provider.model_name,
            messages=cast(Iterable[ChatCompletionMessageParam], message_dicts),
            temperature=temperature,
            tools=(
                cast(Iterable[ChatCompletionToolParam], tool_dicts)
                if tool_dicts
                else openai.NotGiven()
            ),
            max_tokens=max_tokens or openai.NotGiven(),
        )

        openai_message = completion.choices[0].message
        if openai_message.refusal:
            raise Exception(completion.choices[0].message.refusal)

        message = Message(
            content=openai_message.content,
            role=openai_message.role,
            tool_calls=(
                [
                    ToolCall(id=call.id, function=call.function.name, args=call.function.arguments)
                    for call in openai_message.tool_calls
                ]
                if openai_message.tool_calls
                else None
            ),
        )

        usage = Usage(
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
        )

        return LlmGenerateTextResponse(
            message=message,
            metadata=LlmResponseMetadata(
                model=self.provider.model_name,
                provider_name=self.provider.provider_name,
                usage=usage,
            ),
        )

    def generate_structured(
        self,
        *,
        prompt: str,
        messages: list[Message] | None = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = [],
        temperature: float | None = None,
        response_format: Type[StructuredOutputType],
        max_tokens: int | None = None,
    ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
        message_dicts, tool_dicts = self._prep_message_and_tools(
            messages=messages,
            prompt=prompt,
            system_prompt=system_prompt,
            tools=tools,
        )

        openai_client = self.get_client()

        completion = openai_client.beta.chat.completions.parse(
            model=self.provider.model_name,
            messages=cast(Iterable[ChatCompletionMessageParam], message_dicts),
            temperature=temperature,
            tools=(
                cast(Iterable[ChatCompletionToolParam], tool_dicts)
                if tool_dicts
                else openai.NotGiven()
            ),
            response_format=response_format,  # type: ignore
            max_tokens=max_tokens or openai.NotGiven(),
        )

        openai_message = completion.choices[0].message
        if openai_message.refusal:
            raise Exception(completion.choices[0].message.refusal)

        parsed = cast(StructuredOutputType, completion.choices[0].message.parsed)

        usage = Usage(
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
        )

        return LlmGenerateStructuredResponse(
            parsed=parsed,
            metadata=LlmResponseMetadata(
                model=self.provider.model_name,
                provider_name=self.provider.provider_name,
                usage=usage,
            ),
        )

    @staticmethod
    def _prep_message_and_tools(
        *,
        messages: list[Message] | None = None,
        prompt: str | None = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = [],
    ):
        message_dicts = (
            [message.to_message(LlmProviderType.OPENAI) for message in messages] if messages else []
        )
        if system_prompt:
            message_dicts.insert(
                0, Message(role="system", content=system_prompt).to_message(LlmProviderType.OPENAI)
            )
        if prompt:
            message_dicts.insert(
                0, Message(role="user", content=prompt).to_message(LlmProviderType.OPENAI)
            )

        tool_dicts = (
            [tool.to_dict(LlmProviderType.OPENAI) for tool in tools]
            if tools and len(tools) > 0
            else None
        )

        return message_dicts, tool_dicts


@dataclass
class AnthropicProvider:
    provider: LlmProviderDefinition

    default_configs = [
        {
            "match": r".*",
            "temperature": 0.0,
        },
    ]

    @staticmethod
    @inject
    def get_client(app_config: AppConfig = injected) -> anthropic.AnthropicVertex:
        return anthropic.AnthropicVertex(
            project_id=app_config.GOOGLE_CLOUD_PROJECT,
            region="europe-west1" if app_config.USE_EU_REGION else "us-east5",
        )

    @classmethod
    def model(cls, model_name: str) -> "AnthropicProvider":
        return cls(
            provider=LlmProviderDefinition(
                model_name=model_name, provider_name=LlmProviderType.ANTHROPIC
            )
        )

    def _get_config(self, model_name: str):
        for config in self.default_configs:
            if re.match(config["match"], model_name):
                return config
        return None

    @observe(as_type="generation", name="Anthropic Generation")
    @inject
    def generate_text(
        self,
        *,
        system_prompt: str | None = None,
        message_dicts: Iterable[MessageParam],
        tool_dicts: Iterable[ToolParam] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        anthropic_client = self.get_client()

        config = self._get_config(self.provider.model_name)
        if config:
            if temperature is None:
                temperature = config["temperature"]

        completion = anthropic_client.messages.create(
            system=system_prompt or NOT_GIVEN,
            model=self.provider.model_name,
            tools=tool_dicts or NOT_GIVEN,
            messages=message_dicts,
            max_tokens=max_tokens or 8192,
            temperature=temperature or NOT_GIVEN,
        )

        message = self._format_claude_response_to_message(completion)

        usage = Usage(
            completion_tokens=completion.usage.output_tokens,
            prompt_tokens=completion.usage.input_tokens,
            total_tokens=completion.usage.input_tokens + completion.usage.output_tokens,
        )

        langfuse_context.update_current_observation(model=self.provider.model_name, usage=usage)

        return LlmGenerateTextResponse(
            message=message,
            metadata=LlmResponseMetadata(
                model=self.provider.model_name,
                provider_name=self.provider.provider_name,
                usage=usage,
            ),
        )

    @staticmethod
    def _format_claude_response_to_message(completion: anthropic.types.Message) -> Message:
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

    @staticmethod
    def _prep_message_and_tools(
        *,
        messages: list[Message] | None = None,
        prompt: str | None = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = [],
    ):
        message_dicts = (
            [message.to_message(LlmProviderType.ANTHROPIC) for message in messages]
            if messages
            else []
        )
        if prompt:
            message_dicts.insert(
                0, Message(role="user", content=prompt).to_message(LlmProviderType.ANTHROPIC)
            )

        tool_dicts = (
            [tool.to_dict(LlmProviderType.ANTHROPIC) for tool in tools]
            if tools and len(tools) > 0
            else None
        )

        return message_dicts, tool_dicts, system_prompt


LlmProvider = Union[OpenAiProvider, AnthropicProvider]


class LlmClient:
    @observe(name="Generate Text")
    def generate_text(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        model: LlmProvider,
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionTool]] = [],
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_name: str | None = None,
    ) -> LlmGenerateTextResponse:
        if run_name:
            langfuse_context.update_current_observation(name=run_name + " - Generate Text")
            langfuse_context.flush()

        if model.provider.provider_name == LlmProviderType.OPENAI:
            model = cast(OpenAiProvider, model)
            message_dicts, tool_dicts = model._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
            )
            return model.generate_text(
                message_dicts=cast(Iterable[ChatCompletionMessageParam], message_dicts),
                tool_dicts=(
                    cast(Iterable[ChatCompletionToolParam], tool_dicts) if tool_dicts else None
                ),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif model.provider.provider_name == LlmProviderType.ANTHROPIC:
            model = cast(AnthropicProvider, model)
            message_dicts, tool_dicts, system_prompt = model._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
            )
            return model.generate_text(
                system_prompt=system_prompt,
                message_dicts=cast(Iterable[MessageParam], message_dicts),
                tool_dicts=cast(Iterable[ToolParam], tool_dicts) if tool_dicts else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Invalid provider: {model.provider.provider_name}")

    @observe(name="Generate Structured")
    def generate_structured(
        self,
        *,
        prompt: str,
        messages: list[Message] | None = None,
        model: LlmProvider,
        system_prompt: Optional[str] = None,
        response_format: Type[StructuredOutputType],
        tools: Optional[list[FunctionTool]] = [],
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_name: str | None = None,
    ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
        if run_name:
            langfuse_context.update_current_observation(name=run_name + " - Generate Structured")

        if model.provider.provider_name == LlmProviderType.OPENAI:
            model = cast(OpenAiProvider, model)
            return model.generate_structured(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                temperature=temperature,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        elif model.provider.provider_name == LlmProviderType.ANTHROPIC:
            raise NotImplementedError("Anthropic structured outputs are not yet supported")
        else:
            raise ValueError(f"Invalid provider: {model.provider.provider_name}")

    @staticmethod
    def clean_tool_call_assistant_messages(messages: list[Message]) -> list[Message]:
        new_messages = []
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                new_messages.append(
                    Message(role="assistant", content=message.content, tool_calls=[])
                )
            elif message.role == "tool":
                new_messages.append(Message(role="user", content=message.content, tool_calls=[]))
            elif message.role == "tool_use":
                new_messages.append(
                    Message(role="assistant", content=message.content, tool_calls=[])
                )
            else:
                new_messages.append(message)
        return new_messages


@module.provider
def provide_llm_client() -> LlmClient:
    return LlmClient()
