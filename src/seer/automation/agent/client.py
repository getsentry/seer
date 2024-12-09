import json
import logging
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Type, Union, cast

import anthropic
from anthropic import NOT_GIVEN
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmGenerateTextResponse,
    LlmModelDefaultConfig,
    LlmProviderDefaults,
    LlmProviderType,
    LlmRefusalError,
    LlmResponseMetadata,
    Message,
    StructuredOutputType,
    ToolCall,
    Usage,
)
from seer.automation.agent.tools import FunctionTool
from seer.bootup import module
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)

# Token management constants
MAX_TOTAL_TOKENS = 8192     # Maximum total tokens allowed
MAX_PROMPT_TOKENS = 6144    # Maximum prompt tokens allowed 
MAX_COMPLETION_TOKENS = 2048 # Maximum completion tokens allowed
DEFAULT_MAX_MESSAGES = 10   # Default maximum number of messages to keep in history

@dataclass
class OpenAiProvider:
    model_name: str
    provider_name = LlmProviderType.OPENAI
    defaults: LlmProviderDefaults | None = None

    default_configs: ClassVar[list[LlmModelDefaultConfig]] = [
        LlmModelDefaultConfig(
            match=r"^o1-mini.*",
            defaults=LlmProviderDefaults(temperature=1.0),
        ),
        LlmModelDefaultConfig(
            match=r"^o1-preview.*",
            defaults=LlmProviderDefaults(temperature=1.0),
        ),
        LlmModelDefaultConfig(
            match=r".*",
            defaults=LlmProviderDefaults(temperature=0.0),
        ),
    ]

    @staticmethod
    def get_client() -> openai.Client:
        return openai.Client()

    @classmethod
    def model(cls, model_name: str) -> "OpenAiProvider":
        model_config = cls._get_config(model_name)
        return cls(
            model_name=model_name,
            defaults=model_config.defaults if model_config else None,
        )

    @classmethod
    def _get_config(cls, model_name: str):
        for config in cls.default_configs:
            if re.match(config.match, model_name):
                return config
        return None

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ):
        message_dicts, tool_dicts = self._prep_message_and_tools(
            messages=messages,
            prompt=prompt,
            system_prompt=system_prompt,
            tools=tools,
        )

        openai_client = self.get_client()

        completion = openai_client.chat.completions.create(
            model=self.model_name,
            messages=cast(Iterable[ChatCompletionMessageParam], message_dicts),
            temperature=temperature,
            tools=(
                cast(Iterable[ChatCompletionToolParam], tool_dicts)
                if tool_dicts
                else openai.NotGiven()
            ),
            max_tokens=max_tokens or openai.NotGiven(),
            timeout=timeout or openai.NotGiven(),
        )

        openai_message = completion.choices[0].message
        if openai_message.refusal:
            raise LlmRefusalError(completion.choices[0].message.refusal)

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
                model=self.model_name,
                provider_name=self.provider_name,
                usage=usage,
            ),
        )

    def generate_structured(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        response_format: Type[StructuredOutputType],
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
        message_dicts, tool_dicts = self._prep_message_and_tools(
            messages=messages,
            prompt=prompt,
            system_prompt=system_prompt,
            tools=tools,
        )

        openai_client = self.get_client()

        completion = openai_client.beta.chat.completions.parse(
            model=self.model_name,
            messages=cast(Iterable[ChatCompletionMessageParam], message_dicts),
            temperature=temperature,
            tools=(
                cast(Iterable[ChatCompletionToolParam], tool_dicts)
                if tool_dicts
                else openai.NotGiven()
            ),
            response_format=response_format,
            max_tokens=max_tokens or openai.NotGiven(),
            timeout=timeout or openai.NotGiven(),
        )

        openai_message = completion.choices[0].message
        if openai_message.refusal:
            raise LlmRefusalError(completion.choices[0].message.refusal)

        parsed = cast(StructuredOutputType, completion.choices[0].message.parsed)

        usage = Usage(
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
        )

        return LlmGenerateStructuredResponse(
            parsed=parsed,
            metadata=LlmResponseMetadata(
                model=self.model_name,
                provider_name=self.provider_name,
                usage=usage,
            ),
        )

    @staticmethod
    def to_message_dict(message: Message) -> ChatCompletionMessageParam:
        message_dict: dict[str, Any] = {
            "content": message.content if message.content else "",
            "role": message.role,
        }

        if message.tool_calls:
            tool_calls = [tool_call.model_dump(mode="json") for tool_call in message.tool_calls]
            parsed_tool_calls = []
            for item in tool_calls:
                new_item = item.copy()
                new_item["function"] = {"name": item["function"], "arguments": item["args"]}
                new_item["type"] = "function"
                parsed_tool_calls.append(new_item)
            message_dict["tool_calls"] = parsed_tool_calls

        if message.tool_call_id:
            message_dict["tool_call_id"] = message.tool_call_id

        return cast(ChatCompletionMessageParam, message_dict)

    @staticmethod
    def to_tool_dict(tool: FunctionTool) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: {
                            key: value
                            for key, value in {
                                "type": param["type"],
                                "description": param.get("description", ""),
                                "items": param.get("items"),
                            }.items()
                            if value is not None
                        }
                        for param in tool.parameters
                    },
                    "required": tool.required,
                },
            },
        )

    @classmethod
    def _prep_message_and_tools(
        cls,
        *,
        messages: list[Message] | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
    ):
        message_dicts = [cls.to_message_dict(message) for message in messages] if messages else []
        if system_prompt:
            message_dicts.insert(
                0, cls.to_message_dict(Message(role="system", content=system_prompt))
            )
        if prompt:
            message_dicts.append(cls.to_message_dict(Message(role="user", content=prompt)))

        tool_dicts = (
            [cls.to_tool_dict(tool) for tool in tools] if tools and len(tools) > 0 else None
        )

        return message_dicts, tool_dicts


@dataclass
class AnthropicProvider:
    model_name: str
    provider_name = LlmProviderType.ANTHROPIC
    defaults: LlmProviderDefaults | None = None

    default_configs: ClassVar[list[LlmModelDefaultConfig]] = [
        LlmModelDefaultConfig(
            match=r".*",
            defaults=LlmProviderDefaults(temperature=0.0),
        ),
    ]

    @staticmethod
    @inject
    def get_client(app_config: AppConfig = injected) -> anthropic.AnthropicVertex:
        return anthropic.AnthropicVertex(
            project_id=app_config.GOOGLE_CLOUD_PROJECT,
            region="europe-west1" if app_config.USE_EU_REGION else "us-east5",
            max_retries=8,
        )

    @classmethod
    def model(cls, model_name: str) -> "AnthropicProvider":
        model_config = cls._get_config(model_name)
        return cls(
            model_name=model_name,
            defaults=model_config.defaults if model_config else None,
        )

    @classmethod
    def _get_config(cls, model_name: str):
        for config in cls.default_configs:
            if re.match(config.match, model_name):
                return config
        return None

    @observe(as_type="generation", name="Anthropic Generation")
    @inject
    def generate_text(
        self,
        *,
        messages: list[Message] | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ):
        message_dicts, tool_dicts, system_prompt = self._prep_message_and_tools(
            messages=messages,
            prompt=prompt,
            system_prompt=system_prompt,
            tools=tools,
        )

        anthropic_client = self.get_client()

        completion = anthropic_client.messages.create(
            system=system_prompt or NOT_GIVEN,
            model=self.model_name,
            tools=cast(Iterable[ToolParam], tool_dicts) if tool_dicts else NOT_GIVEN,
            messages=cast(Iterable[MessageParam], message_dicts),
            max_tokens=max_tokens or 8192,
            temperature=temperature or NOT_GIVEN,
            timeout=timeout or NOT_GIVEN,
        )

        message = self._format_claude_response_to_message(completion)

        usage = Usage(
            completion_tokens=completion.usage.output_tokens,
            prompt_tokens=completion.usage.input_tokens,
            total_tokens=completion.usage.input_tokens + completion.usage.output_tokens,
        )

        langfuse_context.update_current_observation(model=self.model_name, usage=usage)

        return LlmGenerateTextResponse(
            message=message,
            metadata=LlmResponseMetadata(
                model=self.model_name,
                provider_name=self.provider_name,
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
    def to_message_param(message: Message) -> MessageParam:
        if message.role == "tool":
            return MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        type="tool_result",
                        content=message.content or "",
                        tool_use_id=message.tool_call_id or "",
                    )
                ],
            )
        elif message.role == "tool_use":
            if not message.tool_calls:
                return MessageParam(role="assistant", content=[])
            tool_call = message.tool_calls[0]  # Assuming only one tool call per message
            return MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        type="tool_use",
                        id=tool_call.id or "",
                        name=tool_call.function,
                        input=json.loads(tool_call.args),
                    )
                ],
            )
        else:
            return MessageParam(
                role=message.role,  # type: ignore
                content=[TextBlockParam(type="text", text=message.content or "")],
            )

    @staticmethod
    def to_tool_dict(tool: FunctionTool) -> ToolParam:
        return ToolParam(
            name=tool.name,
            description=tool.description,
            input_schema={
                "type": "object",
                "properties": {
                    param["name"]: {
                        key: value
                        for key, value in {
                            "type": param["type"],
                            "description": param.get("description", ""),
                            "items": param.get("items"),
                        }.items()
                        if value is not None
                    }
                    for param in tool.parameters
                },
                "required": tool.required,
            },
        )

    @classmethod
    def _prep_message_and_tools(
        cls,
        *,
        messages: list[Message] | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
    ) -> tuple[list[MessageParam], list[ToolParam] | None, str | None]:
        message_dicts = [cls.to_message_param(message) for message in messages] if messages else []
        if prompt:
            message_dicts.append(cls.to_message_param(Message(role="user", content=prompt)))

        tool_dicts = (
            [cls.to_tool_dict(tool) for tool in tools] if tools and len(tools) > 0 else None
        )

        return message_dicts, tool_dicts, system_prompt



LlmProvider = Union[OpenAiProvider, AnthropicProvider]

class TokenManager:
    @staticmethod
    def estimate_tokens(messages: List[Message]) -> int:
        """
        Estimates token count for messages. This is a simplified estimation - 
        assuming ~4 chars per token as a rough estimate.
        """
        total_chars = sum(
            len(msg.content or "") + 
            sum(len(str(call.args or "")) for call in (msg.tool_calls or []))
            for msg in messages
        )
        return total_chars // 4  # Rough estimation of tokens

    @staticmethod
    def truncate_messages(
        messages: List[Message], 
        max_tokens: int = MAX_PROMPT_TOKENS,
        max_messages: int = DEFAULT_MAX_MESSAGES
    ) -> List[Message]:
        """
        Truncates message history to fit within token limits while preserving context.
        Keeps system messages and most recent messages.
        """
        if not messages:
            return []

        # Always keep system messages
        system_messages = [m for m in messages if m.role == "system"]
        other_messages = [m for m in messages if m.role != "system"]

        # Keep most recent messages up to max_messages
        truncated = other_messages[-max_messages:] if len(other_messages) > max_messages else other_messages
        
        # Combine system messages with truncated messages
        result = system_messages + truncated

        # Check token count and truncate further if needed
        while TokenManager.estimate_tokens(result) > max_tokens and len(result) > len(system_messages) + 1:
            # Remove the oldest non-system message
            for i, msg in enumerate(result):
                if msg.role != "system":
                    result.pop(i)
                    break

        return result

    @staticmethod
    def validate_token_limits(
        messages: List[Message], 
        max_tokens: int | None = None
    ) -> tuple[List[Message], int]:
        """
        Validates and adjusts message history and completion tokens to stay within limits.
        """
        estimated_prompt_tokens = TokenManager.estimate_tokens(messages)
        
        if estimated_prompt_tokens > MAX_PROMPT_TOKENS:
            messages = TokenManager.truncate_messages(messages, MAX_PROMPT_TOKENS)
            estimated_prompt_tokens = TokenManager.estimate_tokens(messages)

        safe_max_tokens = min(MAX_TOTAL_TOKENS - estimated_prompt_tokens, MAX_COMPLETION_TOKENS, max_tokens or MAX_COMPLETION_TOKENS)
        return messages, safe_max_tokens

class LlmClient:
    @observe(name="Generate Text")
    def generate_text(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        model: LlmProvider,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_name: str | None = None,
        timeout: float | None = None,
    ) -> LlmGenerateTextResponse:
        try:
            if run_name:
                langfuse_context.update_current_observation(name=run_name + " - Generate Text")
                langfuse_context.flush()

            defaults = model.defaults
            default_temperature = defaults.temperature if defaults else None

            messages = LlmClient.clean_message_content(messages if messages else [])
            if not tools:
                messages = LlmClient.clean_tool_call_assistant_messages(messages)

            # Add token management
            messages, adjusted_max_tokens = TokenManager.validate_token_limits(messages, max_tokens)

            if model.provider_name == LlmProviderType.OPENAI:
                model = cast(OpenAiProvider, model)
                return model.generate_text(
                    max_tokens=adjusted_max_tokens,
                    messages=messages,
                    messages=messages,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature or default_temperature,
                    tools=tools,
                    timeout=timeout,
                )
            elif model.provider_name == LlmProviderType.ANTHROPIC:
                model = cast(AnthropicProvider, model)
                return model.generate_text(
                    max_tokens=adjusted_max_tokens,
                    messages=messages,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature or default_temperature,
                    tools=tools,
                    timeout=timeout,
                )
            else:
                raise ValueError(f"Invalid provider: {model.provider_name}")
        except Exception as e:
            logger.exception(f"Text generation failed with provider {model.provider_name}: {e}")
            raise e

    @observe(name="Generate Structured")
    def generate_structured(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        model: LlmProvider,
        system_prompt: str | None = None,
        response_format: Type[StructuredOutputType],
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_name: str | None = None,
        timeout: float | None = None,
    ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
        try:
            if run_name:
                langfuse_context.update_current_observation(
                    name=run_name + " - Generate Structured"
                )

            messages = LlmClient.clean_message_content(messages if messages else [])
            messages = LlmClient.clean_tool_call_assistant_messages(messages)

            # Add token management
            messages, adjusted_max_tokens = TokenManager.validate_token_limits(messages, max_tokens)

            if model.provider_name == LlmProviderType.OPENAI:
                model = cast(OpenAiProvider, model)
                return model.generate_structured(
                    max_tokens=adjusted_max_tokens,
                    messages=messages,
                    messages=messages,
                    prompt=prompt,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    tools=tools,
                    timeout=timeout,
                )
            elif model.provider_name == LlmProviderType.ANTHROPIC:
                raise NotImplementedError("Anthropic structured outputs are not yet supported")
            else:
                raise ValueError(f"Invalid provider: {model.provider_name}")
        except Exception as e:
            logger.exception(f"Text generation failed with provider {model.provider_name}: {e}")
            raise e

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

    @staticmethod
    def clean_message_content(messages: list[Message]) -> list[Message]:
        new_messages = []
        for message in messages:
            # Only add messages that have actual content or tool calls
            if message.content or message.tool_calls:
                new_messages.append(message)
        return new_messages


@module.provider
def provide_llm_client() -> LlmClient:
    return LlmClient()
