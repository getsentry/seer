import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Iterator, Tuple, Type, TypeVar, Union, cast

import anthropic
import sentry_sdk
from anthropic import NOT_GIVEN
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ThinkingBlockParam,
    ThinkingConfigEnabledParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from google import genai  # type: ignore[attr-defined]
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    Content,
    CreateCachedContentConfig,
    FunctionDeclaration,
    GenerateContentConfig,
    GenerateContentResponse,
    GoogleSearch,
    Part,
    ThinkingConfig,
)
from google.genai.types import Tool as GeminiTool
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from requests.exceptions import ChunkedEncodingError

from seer.automation.agent.models import (
    DEFAULT_FIRST_TOKEN_TIMEOUT,
    DEFAULT_INACTIVITY_TIMEOUT,
    LlmGenerateStructuredResponse,
    LlmGenerateTextResponse,
    LlmModelDefaultConfig,
    LlmNoCompletionTokensError,
    LlmProviderDefaults,
    LlmProviderType,
    LlmRefusalError,
    LlmResponseMetadata,
    LlmStreamFirstTokenTimeoutError,
    LlmStreamInactivityTimeoutError,
    LlmStreamTimeoutError,
    Message,
    ResolvedParameters,
    StreamResetException,
    StructuredOutputType,
    ToolCall,
    Usage,
)
from seer.automation.agent.tools import ClaudeTool, FunctionTool
from seer.bootup import module
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected
from seer.utils import backoff_on_exception, backoff_on_generator

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_BACKOFF_MAX_TRIES = 3


@dataclass
class BaseLlmProvider:
    model_name: str
    defaults: LlmProviderDefaults | None = None
    region: str | None = None
    backoff_max_tries: int = DEFAULT_BACKOFF_MAX_TRIES
    sleep_sec_scaler: Callable[[int], float] | None = None

    default_configs: ClassVar[list[LlmModelDefaultConfig]]

    @classmethod
    def get_config(cls, model_name: str) -> LlmModelDefaultConfig | None:
        for config in cls.default_configs:
            if re.match(config.match, model_name):
                return config
        return None

    @inject
    def get_region_preference(self, app_config: AppConfig = injected) -> list[str] | None:
        config = self.get_config(self.model_name)
        if config is None:
            return None

        if config.region_preference is None:
            return None

        if app_config.SENTRY_REGION == "de":
            return config.region_preference["de"]

        if app_config.SENTRY_REGION in config.region_preference:
            return config.region_preference[app_config.SENTRY_REGION]

        if "*" in config.region_preference:
            return config.region_preference["*"]

        # All regions except DE will use the US region preference if no generic preference is available.
        if "us" in config.region_preference:
            return config.region_preference["us"]

        # If no region preference is available, return None
        return None

    def copy(self, **kwargs):
        params = {
            "model_name": self.model_name,
            "defaults": self.defaults,
            "region": self.region,
            "backoff_max_tries": self.backoff_max_tries,
            "sleep_sec_scaler": self.sleep_sec_scaler,
        }
        params.update(kwargs)
        return type(self)(**params)


def _iterate_with_timeouts(
    stream_generator: Iterator[T],
    first_token_timeout: float,
    inactivity_timeout: float,
    on_cleanup: Callable[[], None] | None = None,
) -> Iterator[T]:
    """Helper to iterate over a blocking stream with timeouts for first token and inactivity.

    Args:
        stream_generator: The stream to iterate over
        first_token_timeout: Timeout in seconds for receiving the first token
        inactivity_timeout: Timeout in seconds between subsequent tokens
        on_cleanup: Optional callback to clean up resources when done
    """
    cancel_event = threading.Event()
    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    last_yield_time = time.time()
    first_token_received = False

    def producer():
        try:
            for item in stream_generator:
                if cancel_event.is_set():
                    break
                q.put(("data", item))
            q.put(("end", None))
        except Exception as e:
            q.put(("error", e))
        finally:
            if on_cleanup:
                try:
                    on_cleanup()
                except Exception:
                    pass

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    try:
        while True:
            timeout_to_use = first_token_timeout if not first_token_received else inactivity_timeout
            try:
                msg_type, item = q.get(timeout=timeout_to_use)
            except queue.Empty:
                cancel_event.set()
                if first_token_received:
                    raise LlmStreamInactivityTimeoutError(
                        f"Stream inactivity timeout after {timeout_to_use} seconds"
                    )
                else:
                    raise LlmStreamFirstTokenTimeoutError(
                        f"Stream time to first token timeout after {timeout_to_use} seconds"
                    )

            if msg_type == "data":
                first_token_received = True
                last_yield_time = time.time()
                if time.time() - last_yield_time > inactivity_timeout:
                    raise LlmStreamInactivityTimeoutError(
                        f"Stream inactivity timeout after {timeout_to_use} seconds"
                    )
                yield item
            elif msg_type == "error":
                raise item
            elif msg_type == "end":
                break
    finally:
        cancel_event.set()


@dataclass
class OpenAiProvider(BaseLlmProvider):
    provider_name = LlmProviderType.OPENAI

    default_configs: ClassVar[list[LlmModelDefaultConfig]] = [
        LlmModelDefaultConfig(
            match=r"^o.*",
            defaults=LlmProviderDefaults(temperature=1.0, first_token_timeout=90.0),
        ),
        LlmModelDefaultConfig(
            match=r".*",
            defaults=LlmProviderDefaults(temperature=0.0),
        ),
    ]

    @staticmethod
    def is_completion_exception_retryable(exception: Exception) -> bool:
        return isinstance(exception, openai.InternalServerError) or isinstance(
            exception, LlmStreamTimeoutError
        )

    @inject
    def get_client(self, app_config: AppConfig = injected) -> openai.Client:
        if app_config.SENTRY_REGION == "de":
            raise ValueError("OpenAI is not available in de")
        if self.region:
            raise ValueError("Cannot set region for OpenAiProvider")

        client = openai.Client(max_retries=4)

        return client

    @classmethod
    def model(
        cls,
        model_name: str,
        region: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        seed: int | None = None,
    ) -> "OpenAiProvider":
        model_config = cls.get_config(model_name)

        # Merge model-level config with default config
        base_defaults = model_config.defaults if model_config else None

        # Create new defaults that override base defaults with model-level config
        merged_defaults = LlmProviderDefaults(
            temperature=(
                temperature
                if temperature is not None
                else (base_defaults.temperature if base_defaults else None)
            ),
            max_tokens=(
                max_tokens
                if max_tokens is not None
                else (base_defaults.max_tokens if base_defaults else None)
            ),
            reasoning_effort=(
                reasoning_effort
                if reasoning_effort is not None
                else (base_defaults.reasoning_effort if base_defaults else None)
            ),
            seed=seed if seed is not None else (base_defaults.seed if base_defaults else None),
        )

        return cls(
            model_name=model_name,
            defaults=merged_defaults,
            region=region,
        )

    @staticmethod
    def is_input_too_long(exception: Exception) -> bool:
        return isinstance(exception, openai.BadRequestError) and "context_length_exceeded" in str(
            exception
        )

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        predicted_output: str | None = None,
        reasoning_effort: str | None = None,
    ):
        @sentry_sdk.trace
        @backoff_on_exception(
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )
        def _generate_text():
            message_dicts, tool_dicts = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )

            openai_client = self.get_client()

            completion = openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(Iterable[ChatCompletionMessageParam], message_dicts),
                temperature=temperature,
                seed=seed,
                tools=(
                    cast(Iterable[ChatCompletionToolParam], tool_dicts)
                    if tool_dicts
                    else openai.NotGiven()
                ),
                max_tokens=max_tokens or openai.NotGiven(),
                timeout=timeout or openai.NotGiven(),
                prediction=(
                    {
                        "type": "content",
                        "content": predicted_output,
                    }
                    if predicted_output
                    else openai.NotGiven()
                ),
                reasoning_effort=reasoning_effort if reasoning_effort else openai.NotGiven(),
            )

            openai_message = completion.choices[0].message
            if openai_message.refusal:
                raise LlmRefusalError(completion.choices[0].message.refusal)

            message = Message(
                content=openai_message.content,
                role=openai_message.role,
                tool_calls=(
                    [
                        ToolCall(
                            id=call.id, function=call.function.name, args=call.function.arguments
                        )
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

        return _generate_text()

    def generate_structured(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        response_format: Type[StructuredOutputType],
        max_tokens: int | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
    ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
        @sentry_sdk.trace
        @backoff_on_exception(
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )
        def _generate_structured():
            message_dicts, tool_dicts = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )

            openai_client = self.get_client()

            completion = openai_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=cast(Iterable[ChatCompletionMessageParam], message_dicts),
                temperature=temperature,
                seed=seed,
                tools=(
                    cast(Iterable[ChatCompletionToolParam], tool_dicts)
                    if tool_dicts
                    else openai.NotGiven()
                ),
                response_format=response_format,
                max_tokens=max_tokens or openai.NotGiven(),
                timeout=timeout or openai.NotGiven(),
                reasoning_effort=reasoning_effort if reasoning_effort else openai.NotGiven(),
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

        return _generate_structured()

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
            message_dict["role"] = "assistant"

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
        reasoning_effort: str | None = None,
    ):
        message_dicts = [cls.to_message_dict(message) for message in messages] if messages else []
        if system_prompt:
            message_dicts.insert(
                0,
                cls.to_message_dict(
                    Message(
                        role="system" if not reasoning_effort else "developer",
                        content=system_prompt,
                    )
                ),
            )
        if prompt:
            message_dicts.append(cls.to_message_dict(Message(role="user", content=prompt)))

        tool_dicts = (
            [cls.to_tool_dict(tool) for tool in tools] if tools and len(tools) > 0 else None
        )

        return message_dicts, tool_dicts

    def generate_text_stream(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        first_token_timeout: float,
        inactivity_timeout: float,
    ) -> Iterator[Tuple[str, str] | ToolCall | Usage | StreamResetException]:
        @observe(as_type="generation", name="OpenAI Stream")
        @sentry_sdk.trace
        def _stream_with_retries():
            message_dicts, tool_dicts = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )

            openai_client = self.get_client()

            stream = openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(Iterable[ChatCompletionMessageParam], message_dicts),
                temperature=temperature,
                seed=seed,
                tools=(
                    cast(Iterable[ChatCompletionToolParam], tool_dicts)
                    if tool_dicts
                    else openai.NotGiven()
                ),
                max_tokens=max_tokens or openai.NotGiven(),
                timeout=timeout or openai.NotGiven(),
                stream=True,
                stream_options={"include_usage": True},
                reasoning_effort=reasoning_effort if reasoning_effort else openai.NotGiven(),
            )

            current_tool_call: dict[str, Any] | None = None
            current_tool_call_index = 0

            def cleanup():
                try:
                    stream.response.close()
                except Exception:
                    pass

            for chunk in _iterate_with_timeouts(
                stream,
                first_token_timeout=first_token_timeout,
                inactivity_timeout=inactivity_timeout,
                on_cleanup=cleanup,
            ):
                if not chunk.choices and chunk.usage:
                    usage = Usage(
                        completion_tokens=chunk.usage.completion_tokens,
                        prompt_tokens=chunk.usage.prompt_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                    yield usage
                    langfuse_context.update_current_observation(model=self.model_name, usage=usage)
                    break

                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    tool_call = delta.tool_calls[0]

                    if (
                        not current_tool_call or current_tool_call_index != tool_call.index
                    ):  # Start of new tool call
                        current_tool_call_index = tool_call.index
                        if current_tool_call:
                            yield ToolCall(**current_tool_call)
                        current_tool_call = None
                        current_tool_call = {
                            "id": tool_call.id,
                            "function": tool_call.function.name if tool_call.function.name else "",
                            "args": (
                                tool_call.function.arguments if tool_call.function.arguments else ""
                            ),
                        }
                    else:
                        if tool_call.function.arguments:
                            current_tool_call["args"] += tool_call.function.arguments
                if chunk.choices[0].finish_reason == "tool_calls" and current_tool_call:
                    yield ToolCall(**current_tool_call)
                if delta.content:
                    yield "content", delta.content

        return backoff_on_generator(
            _stream_with_retries,
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )()

    def construct_message_from_stream(
        self, content_chunks: list[str], tool_calls: list[ToolCall]
    ) -> Message:
        return Message(
            role="assistant",
            content="".join(content_chunks) if content_chunks else None,
            tool_calls=tool_calls if tool_calls else None,
        )


@dataclass
class AnthropicProvider(BaseLlmProvider):
    provider_name = LlmProviderType.ANTHROPIC

    default_configs: ClassVar[list[LlmModelDefaultConfig]] = [
        LlmModelDefaultConfig(
            match=r".*",
            defaults=LlmProviderDefaults(temperature=0.0),
            region_preference={
                "us": ["europe-west4", "global"],
                "de": ["europe-west4"],
                "*": ["global", "us-east5"],
            },
        ),
    ]

    @staticmethod
    def is_completion_exception_retryable(exception: Exception) -> bool:
        retryable_errors = (
            "Internal server error",
            "not_found_error",
            "404, 'message': 'Publisher Model",
        )
        return (
            (
                isinstance(exception, anthropic.AnthropicError)
                and any(error in str(exception) for error in retryable_errors)
            )
            or isinstance(exception, LlmStreamTimeoutError)
            or isinstance(exception, LlmNoCompletionTokensError)
            or "incomplete chunked read" in str(exception)
        )

    @inject
    def get_client(self, app_config: AppConfig = injected) -> anthropic.AnthropicVertex:
        project_id = app_config.GOOGLE_CLOUD_PROJECT
        max_retries = 8

        # Use provided region if available, otherwise fall back to automatic region selection
        if self.region:
            selected_region = self.region

        # Always overwrite
        if app_config.DEV:
            selected_region = "us-east5"

        if not selected_region:
            raise ValueError(
                f"No region selected for model {self.model_name}. AnthropicProvider requires explicit region selection."
            )
        if app_config.SENTRY_REGION == "de" and "europe" not in selected_region:
            raise ValueError(
                f"Cannot route to non europe region {selected_region} for model {self.model_name} while SENTRY_REGION is set to de"
            )

        base_url = None
        if selected_region == "global":
            base_url = "https://aiplatform.googleapis.com/v1/"

        client = anthropic.AnthropicVertex(
            project_id=project_id,
            region=selected_region,
            base_url=base_url,
            max_retries=max_retries,
        )

        return client

    @classmethod
    def model(
        cls,
        model_name: str,
        region: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
    ) -> "AnthropicProvider":
        model_config = cls.get_config(model_name)

        # Merge model-level config with default config
        base_defaults = model_config.defaults if model_config else None

        # Create new defaults that override base defaults with model-level config
        merged_defaults = LlmProviderDefaults(
            temperature=(
                temperature
                if temperature is not None
                else (base_defaults.temperature if base_defaults else None)
            ),
            max_tokens=(
                max_tokens
                if max_tokens is not None
                else (base_defaults.max_tokens if base_defaults else None)
            ),
            reasoning_effort=(
                reasoning_effort
                if reasoning_effort is not None
                else (base_defaults.reasoning_effort if base_defaults else None)
            ),
            timeout=(
                timeout
                if timeout is not None
                else (base_defaults.timeout if base_defaults else None)
            ),
        )

        return cls(
            model_name=model_name,
            defaults=merged_defaults,
            region=region,
        )

    @staticmethod
    def is_input_too_long(exception: Exception) -> bool:
        error_msg = str(exception)
        return ("Prompt is too long" in error_msg) or ("exceed context limit" in error_msg)

    def generate_text(
        self,
        *,
        messages: list[Message] | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool | ClaudeTool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
    ):
        @observe(as_type="generation", name="Anthropic Generation")
        @sentry_sdk.trace
        @backoff_on_exception(
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )
        def _generate_text():
            message_dicts, tool_dicts, system_prompt_block = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
            )

            anthropic_client = self.get_client()

            completion = anthropic_client.messages.create(
                system=system_prompt_block or NOT_GIVEN,
                model=self.model_name,
                tools=cast(Iterable[ToolParam], tool_dicts) if tool_dicts else NOT_GIVEN,
                messages=cast(Iterable[MessageParam], message_dicts),
                max_tokens=max_tokens or 8192,
                temperature=temperature or NOT_GIVEN,
                timeout=timeout or NOT_GIVEN,
                thinking=(
                    ThinkingConfigEnabledParam(
                        type="enabled",
                        budget_tokens=(
                            1024
                            if reasoning_effort == "low"
                            else 4092 if reasoning_effort == "medium" else 8192
                        ),
                    )
                    if reasoning_effort
                    else NOT_GIVEN
                ),
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

        return _generate_text()

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
            elif block.type == "thinking":
                message.thinking_content = block.thinking
                message.thinking_signature = block.signature
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
        elif message.role == "tool_use" or (message.role == "assistant" and message.tool_calls):
            assistant_msg_content: list[ThinkingBlockParam | ToolUseBlockParam] = []
            if message.thinking_content and message.thinking_signature:
                assistant_msg_content.append(
                    ThinkingBlockParam(
                        type="thinking",
                        thinking=message.thinking_content,
                        signature=message.thinking_signature,
                    )
                )
            if message.tool_calls:
                tool_call = message.tool_calls[0]  # Assuming only one tool call per message
                assistant_msg_content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        id=tool_call.id or "",
                        name=tool_call.function,
                        input=json.loads(tool_call.args),
                    )
                )
            return MessageParam(
                role="assistant",
                content=assistant_msg_content,
            )
        else:
            other_content: list[ThinkingBlockParam | TextBlockParam] = []
            if message.thinking_content and message.thinking_signature:
                other_content.append(
                    ThinkingBlockParam(
                        type="thinking",
                        thinking=message.thinking_content,
                        signature=message.thinking_signature,
                    )
                )
            other_content.append(TextBlockParam(type="text", text=message.content or ""))
            return MessageParam(
                role=message.role,  # type: ignore
                content=other_content,
            )

    @staticmethod
    def to_tool_dict(tool: FunctionTool | ClaudeTool) -> ToolParam:
        if isinstance(tool, ClaudeTool):
            return ToolParam(  # type: ignore
                name=tool.name,
                type=tool.type,  # type: ignore
            )

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
        tools: list[FunctionTool | ClaudeTool] | None = None,
    ) -> tuple[list[MessageParam], list[ToolParam] | None, list[TextBlockParam] | None]:
        message_dicts = [cls.to_message_param(message) for message in messages] if messages else []
        if prompt:
            message_dicts.append(cls.to_message_param(Message(role="user", content=prompt)))
        if message_dicts:
            message_dicts[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}  # type: ignore[index]

        tool_dicts = (
            [cls.to_tool_dict(tool) for tool in tools] if tools and len(tools) > 0 else None
        )

        system_prompt_block = (
            [TextBlockParam(type="text", text=system_prompt, cache_control={"type": "ephemeral"})]
            if system_prompt
            else None
        )

        return message_dicts, tool_dicts, system_prompt_block

    def generate_text_stream(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool | ClaudeTool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        first_token_timeout: float,
        inactivity_timeout: float,
    ) -> Iterator[Tuple[str, str] | ToolCall | Usage | StreamResetException]:
        @observe(as_type="generation", name="Anthropic Stream")
        @sentry_sdk.trace
        def _stream_with_retries():
            message_dicts, tool_dicts, system_prompt_block = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
            )

            anthropic_client = self.get_client()

            stream = anthropic_client.messages.create(
                system=system_prompt_block or NOT_GIVEN,
                model=self.model_name,
                tools=cast(Iterable[ToolParam], tool_dicts) if tool_dicts else NOT_GIVEN,
                messages=cast(Iterable[MessageParam], message_dicts),
                max_tokens=max_tokens or 8192,
                temperature=temperature or NOT_GIVEN,
                timeout=timeout or NOT_GIVEN,
                stream=True,
                thinking=(
                    ThinkingConfigEnabledParam(
                        type="enabled",
                        budget_tokens=(
                            1024
                            if reasoning_effort == "low"
                            else 4092 if reasoning_effort == "medium" else 8192
                        ),
                    )
                    if reasoning_effort
                    else NOT_GIVEN
                ),
            )

            try:
                current_tool_call: dict[str, Any] | None = None
                current_input_json = []
                total_input_write_tokens = 0
                total_input_read_tokens = 0
                total_input_tokens = 0
                total_output_tokens = 0

                yielded_content = False

                def cleanup():
                    try:
                        stream.response.close()
                    except Exception:
                        pass

                for chunk in _iterate_with_timeouts(
                    stream,
                    first_token_timeout=first_token_timeout,
                    inactivity_timeout=inactivity_timeout,
                    on_cleanup=cleanup,
                ):
                    if chunk.type == "message_start" and chunk.message.usage:
                        if chunk.message.usage.cache_creation_input_tokens:
                            total_input_write_tokens += (
                                chunk.message.usage.cache_creation_input_tokens
                            )
                        if chunk.message.usage.cache_read_input_tokens:
                            total_input_read_tokens += chunk.message.usage.cache_read_input_tokens
                        total_input_tokens += chunk.message.usage.input_tokens
                        total_output_tokens += chunk.message.usage.output_tokens
                    elif chunk.type == "message_delta" and chunk.usage:
                        total_output_tokens += chunk.usage.output_tokens

                    if chunk.type == "message_stop":
                        break
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        yield "content", chunk.delta.text
                        yielded_content = True
                    elif (
                        chunk.type == "content_block_delta" and chunk.delta.type == "thinking_delta"
                    ):
                        yield "thinking_content", chunk.delta.thinking
                    elif (
                        chunk.type == "content_block_delta"
                        and chunk.delta.type == "signature_delta"
                    ):
                        yield "thinking_signature", chunk.delta.signature
                    elif (
                        chunk.type == "content_block_start"
                        and chunk.content_block.type == "tool_use"
                    ):
                        # Start accumulating a new tool call
                        current_tool_call = {
                            "id": chunk.content_block.id,
                            "function": chunk.content_block.name,
                            "args": "",
                        }
                    elif (
                        chunk.type == "content_block_delta"
                        and chunk.delta.type == "input_json_delta"
                    ):
                        # Accumulate the input JSON
                        if current_tool_call:
                            current_input_json.append(chunk.delta.partial_json)
                    elif chunk.type == "content_block_stop" and current_tool_call:
                        # Tool call is complete, yield it
                        current_tool_call["args"] = "".join(current_input_json)
                        yield ToolCall(**current_tool_call)
                        current_tool_call = None
                        current_input_json = []
                        yielded_content = True

                if not yielded_content or total_output_tokens == 0:
                    raise LlmNoCompletionTokensError("No content returned from Claude")
            finally:
                usage = Usage(
                    completion_tokens=total_output_tokens,
                    prompt_tokens=total_input_tokens,
                    total_tokens=total_input_tokens + total_output_tokens,
                    prompt_cache_write_tokens=total_input_write_tokens,
                    prompt_cache_read_tokens=total_input_read_tokens,
                )
                yield usage
                langfuse_context.update_current_observation(
                    model=self.model_name, usage=usage.to_langfuse_usage()
                )

        return backoff_on_generator(
            _stream_with_retries,
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )()

    def construct_message_from_stream(
        self,
        content_chunks: list[str],
        tool_calls: list[ToolCall],
        thinking_content_chunks: list[str],
        thinking_signature: str | None,
    ) -> Message:
        message = Message(
            role="tool_use" if tool_calls else "assistant",
            content="".join(content_chunks) if content_chunks else None,
            thinking_content="".join(thinking_content_chunks) if thinking_content_chunks else None,
            thinking_signature=thinking_signature,
        )

        if tool_calls:
            message.tool_calls = tool_calls
            message.tool_call_id = tool_calls[0].id

        return message


@dataclass
class GeminiProvider(BaseLlmProvider):
    provider_name = LlmProviderType.GEMINI
    local_regions_only: bool = False

    default_configs: ClassVar[list[LlmModelDefaultConfig]] = [
        LlmModelDefaultConfig(
            match=r".*-preview-.*",  # contains the substring "-preview-"
            defaults=LlmProviderDefaults(temperature=0.0),
            region_preference={
                "us": ["global", "us-central1"],
                # Leave blank for de as preview models are not available in de
            },
        ),
        LlmModelDefaultConfig(
            match=r".*",
            defaults=LlmProviderDefaults(temperature=0.0),
            region_preference={
                "us": ["global", "us-central1", "us-east1"],
                "de": ["europe-west1", "europe-west4"],
            },
        ),
    ]

    @staticmethod
    def is_completion_exception_retryable(exception: Exception) -> bool:
        retryable_errors = (
            # https://sentry.sentry.io/issues/6301072208
            "TLS/SSL connection has been closed",
            "Max retries exceeded with url",
            "Internal error",
            "499 CANCELLED",
        )

        return (
            isinstance(exception, ServerError)
            or (
                isinstance(exception, ClientError)
                and any(error in str(exception) for error in retryable_errors)
            )
            or isinstance(exception, LlmNoCompletionTokensError)
            or isinstance(exception, LlmStreamTimeoutError)
            or isinstance(exception, ChunkedEncodingError)
            or isinstance(exception, json.JSONDecodeError)
        )

    @staticmethod
    def is_input_too_long(exception: Exception) -> bool:
        return isinstance(exception, ClientError) and "input token count" in str(exception)

    @inject
    def get_region_preference(self, app_config: AppConfig = injected) -> list[str] | None:
        region_preference = super().get_region_preference()
        if region_preference and self.local_regions_only:
            return [region for region in region_preference if region != "global"]
        return region_preference

    @inject
    def get_client(self, app_config: AppConfig = injected) -> genai.Client:
        if not self.region:
            raise ValueError(
                f"No region selected for model {self.model_name}. GeminiProvider requires explicit region selection."
            )
        if app_config.SENTRY_REGION == "de" and "europe" not in self.region:
            raise ValueError(
                f"Cannot route to non europe region {self.region} for model {self.model_name} while SENTRY_REGION is set to de"
            )

        client = genai.Client(
            vertexai=True,
            location=self.region,
        )

        return client

    @classmethod
    def model(
        cls,
        model_name: str,
        region: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        local_regions_only: bool = False,
    ) -> "GeminiProvider":
        model_config = cls.get_config(model_name)

        # Merge model-level config with default config
        base_defaults = model_config.defaults if model_config else None

        # Create new defaults that override base defaults with model-level config
        merged_defaults = LlmProviderDefaults(
            temperature=(
                temperature
                if temperature is not None
                else (base_defaults.temperature if base_defaults else None)
            ),
            max_tokens=(
                max_tokens
                if max_tokens is not None
                else (base_defaults.max_tokens if base_defaults else None)
            ),
            seed=seed if seed is not None else (base_defaults.seed if base_defaults else None),
        )

        return cls(
            model_name=model_name,
            defaults=merged_defaults,
            region=region,
            local_regions_only=local_regions_only,
        )

    def search_the_web(
        self, prompt: str, temperature: float | None = None, seed: int | None = None
    ) -> str:
        @observe(as_type="generation", name="Gemini Generation with Grounding")
        @sentry_sdk.trace
        @backoff_on_exception(
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )
        def _search_the_web():
            client = self.get_client()
            google_search_tool = GeminiTool(google_search=GoogleSearch())

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    temperature=temperature or 0.0,
                    seed=seed,
                ),
            )
            answer = ""
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for each in response.candidates[0].content.parts:
                    if each.text:
                        answer += each.text
            return answer

        return _search_the_web()

    def generate_structured(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        response_format: Type[StructuredOutputType],
        max_tokens: int | None = None,
        cache_name: str | None = None,
        thinking_budget: int | None = None,
    ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
        @observe(as_type="generation", name="Gemini Generation")
        @sentry_sdk.trace
        @backoff_on_exception(
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )
        def _generate_structured():
            message_dicts, tool_dicts, final_system_prompt = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
            )

            client = self.get_client()

            max_retries = 2  # Gemini sometimes doesn't fill in response.parsed
            for _ in range(max_retries + 1):
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=message_dicts,  # type: ignore[arg-type]
                    config=GenerateContentConfig(
                        system_instruction=final_system_prompt,
                        tools=tool_dicts,
                        response_modalities=["TEXT"],
                        temperature=temperature or 0.0,
                        seed=seed,
                        response_mime_type="application/json",
                        max_output_tokens=max_tokens or 8192,
                        response_schema=response_format,
                        cached_content=cache_name,
                        thinking_config=(
                            ThinkingConfig(thinking_budget=thinking_budget)
                            if thinking_budget is not None
                            else None
                        ),
                    ),
                )
                if response.parsed is not None:
                    break

            usage = Usage(
                completion_tokens=(
                    response.usage_metadata.candidates_token_count
                    if response.usage_metadata
                    and response.usage_metadata.candidates_token_count is not None
                    else 0
                ),
                prompt_tokens=(
                    response.usage_metadata.prompt_token_count
                    if response.usage_metadata
                    and response.usage_metadata.prompt_token_count is not None
                    else 0
                ),
                total_tokens=(
                    response.usage_metadata.total_token_count
                    if response.usage_metadata
                    and response.usage_metadata.total_token_count is not None
                    else 0
                ),
            )
            langfuse_context.update_current_observation(model=self.model_name, usage=usage)

            return LlmGenerateStructuredResponse(
                parsed=response.parsed,  # type: ignore[arg-type]
                metadata=LlmResponseMetadata(
                    model=self.model_name,
                    provider_name=self.provider_name,
                    usage=usage,
                ),
            )

        return _generate_structured()

    def generate_text_stream(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        first_token_timeout: float,
        inactivity_timeout: float,
    ) -> Iterator[Tuple[str, str] | ToolCall | Usage | StreamResetException]:
        @observe(as_type="generation", name="Gemini Stream")
        @sentry_sdk.trace
        def _stream_with_retries():
            message_dicts, tool_dicts, system_prompt_processed = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
            )

            client = self.get_client()

            total_prompt_tokens = 0
            total_completion_tokens = 0
            output_yielded = False

            try:
                stream = client.models.generate_content_stream(
                    model=self.model_name,
                    contents=message_dicts,  # type: ignore[arg-type]
                    config=GenerateContentConfig(
                        tools=tool_dicts,
                        system_instruction=system_prompt_processed,
                        response_modalities=["TEXT"],
                        temperature=temperature or 0.0,
                        seed=seed,
                        max_output_tokens=max_tokens or 8192,
                        thinking_config=ThinkingConfig(include_thoughts=True),
                    ),
                )

                current_tool_call: dict[str, Any] | None = None

                def cleanup():
                    try:
                        stream.response.close()
                    except Exception:
                        pass

                for chunk in _iterate_with_timeouts(
                    stream,
                    first_token_timeout=first_token_timeout,
                    inactivity_timeout=inactivity_timeout,
                    on_cleanup=cleanup,
                ):

                    # Handle function calls
                    if (
                        chunk.candidates
                        and chunk.candidates[0].content
                        and chunk.candidates[0].content.parts
                        and chunk.candidates[0].content.parts[0].function_call
                    ):
                        function_call = chunk.candidates[0].content.parts[0].function_call
                        if function_call.name and function_call.args and not current_tool_call:
                            current_tool_call = {
                                "id": str(hash(function_call.name + str(function_call.args))),
                                "function": function_call.name,
                                "args": json.dumps(function_call.args),
                            }
                            yield ToolCall(**current_tool_call)
                            output_yielded = True
                            current_tool_call = None
                    # Handle text chunks
                    elif chunk.text is not None:
                        yield "content", str(chunk.text)  # type: ignore[misc]
                        output_yielded = True

                    # Update token counts if available
                    if chunk.usage_metadata:
                        if chunk.usage_metadata.prompt_token_count is not None:
                            total_prompt_tokens = chunk.usage_metadata.prompt_token_count
                        if chunk.usage_metadata.candidates_token_count is not None:
                            total_completion_tokens = chunk.usage_metadata.candidates_token_count

                if not output_yielded:
                    raise LlmNoCompletionTokensError("No output returned from Gemini")
            finally:
                # Yield final usage statistics
                usage = Usage(
                    completion_tokens=total_completion_tokens,
                    prompt_tokens=total_prompt_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                )
                yield usage
                langfuse_context.update_current_observation(model=self.model_name, usage=usage)

        # Apply retry logic to the generator
        retrying_stream = backoff_on_generator(
            _stream_with_retries,
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )

        return retrying_stream()

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ):
        @observe(as_type="generation", name="Gemini Generation")
        @sentry_sdk.trace
        @backoff_on_exception(
            is_exception_retryable=self.is_completion_exception_retryable,
            max_tries=self.backoff_max_tries,
            sleep_sec_scaler=self.sleep_sec_scaler,
        )
        def _generate_text():
            message_dicts, tool_dicts, final_system_prompt = self._prep_message_and_tools(
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
            )

            client = self.get_client()
            response = client.models.generate_content(
                model=self.model_name,
                contents=message_dicts,  # type: ignore[arg-type]
                config=GenerateContentConfig(
                    tools=tool_dicts,
                    system_instruction=final_system_prompt,
                    temperature=temperature or 0.0,
                    seed=seed,
                    max_output_tokens=max_tokens or 8192,
                ),
            )

            message = self._format_gemini_response_to_message(response)

            usage = Usage(
                completion_tokens=(
                    response.usage_metadata.candidates_token_count
                    if response.usage_metadata
                    and response.usage_metadata.candidates_token_count is not None
                    else 0
                ),
                prompt_tokens=(
                    response.usage_metadata.prompt_token_count
                    if response.usage_metadata
                    and response.usage_metadata.prompt_token_count is not None
                    else 0
                ),
                total_tokens=(
                    response.usage_metadata.total_token_count
                    if response.usage_metadata
                    and response.usage_metadata.total_token_count is not None
                    else 0
                ),
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

        return _generate_text()

    @classmethod
    def _prep_message_and_tools(
        cls,
        *,
        messages: list[Message] | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool] | None = None,
    ) -> tuple[list[Content], list[GeminiTool] | None, str | None]:
        contents: list[Content] = []

        if messages:
            # Group consecutive tool messages together
            grouped_messages: list[list[Message]] = []
            current_group: list[Message] = []

            for message in messages:
                if message.role == "tool":
                    current_group.append(message)
                else:
                    if current_group:
                        grouped_messages.append(current_group)
                        current_group = []
                    grouped_messages.append([message])

            if current_group:
                grouped_messages.append(current_group)

            # Convert each group into a Content object
            for group in grouped_messages:
                if len(group) == 1 and group[0].role != "tool":
                    contents.append(cls.to_content(group[0]))
                elif group[0].role == "tool":
                    # Combine multiple tool messages into a single Content
                    parts = [
                        Part.from_function_response(
                            name=msg.tool_call_function or "",
                            response={"response": msg.content},
                        )
                        for msg in group
                    ]
                    contents.append(Content(role="user", parts=parts))

        if prompt:
            contents.append(
                Content(
                    role="user",
                    parts=[Part(text=prompt)],
                )
            )

        processed_tools = [cls.to_tool(tool) for tool in tools] if tools else []

        return contents, processed_tools, system_prompt

    @staticmethod
    def to_content(message: Message) -> Content:
        if message.role == "tool_use" or (message.role == "assistant" and message.tool_calls):
            if not message.tool_calls:
                return Content(
                    role="model",
                    parts=[Part(text=message.content or "")],
                )

            parts = []
            if message.content:
                parts.append(Part(text=message.content))
            for tool_call in message.tool_calls:
                parts.append(
                    Part.from_function_call(
                        name=tool_call.function,
                        args=json.loads(tool_call.args),
                    )
                )
            return Content(role="model", parts=parts)

        elif message.role == "assistant":
            return Content(
                role="model",
                parts=[Part(text=message.content or "")],
            )
        else:
            return Content(
                role="user",
                parts=[Part(text=message.content or "")],
            )

    @staticmethod
    def to_tool(tool: FunctionTool) -> GeminiTool:
        return GeminiTool(
            function_declarations=[
                FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            param["name"]: {
                                key: value
                                for key, value in {
                                    "type": param["type"].upper(),  # type: ignore
                                    "description": param.get("description", ""),
                                    "items": (
                                        {
                                            **param.get("items", {}),  # type: ignore
                                            "type": param.get("items", {}).get("type", "").upper(),  # type: ignore
                                        }
                                        if param.get("items") and "type" in param.get("items", {})
                                        else param.get("items")
                                    ),
                                }.items()
                                if value is not None
                            }
                            for param in tool.parameters
                        },
                        "required": tool.required,
                    },
                )
            ],
        )

    def construct_message_from_stream(
        self, content_chunks: list[str], tool_calls: list[ToolCall]
    ) -> Message:
        message = Message(
            role="tool_use" if tool_calls else "assistant",
            content="".join(content_chunks) if content_chunks else None,
        )

        if tool_calls:
            message.tool_calls = tool_calls
            message.tool_call_id = tool_calls[0].id

        return message

    def _format_gemini_response_to_message(self, response: GenerateContentResponse) -> Message:
        parts = (
            response.candidates[0].content.parts
            if (
                response.candidates
                and len(response.candidates) > 0
                and response.candidates[0].content
                and response.candidates[0].content.parts
            )
            else []
        )

        message = Message(
            role="assistant",
            content=(parts[0].text if parts and parts[0].text else None),
        )

        for part in parts:
            if part.function_call:
                if not message.tool_calls:
                    message.tool_calls = []
                message.tool_calls.append(
                    ToolCall(
                        id=part.function_call.id,
                        function=part.function_call.name or "",
                        args=json.dumps(part.function_call.args),
                    )
                )
                message.role = "tool_use"
                message.tool_call_id = part.function_call.id
            if part.text:
                message.content = part.text

        return message

    @observe(name="Create Gemini cache")
    @sentry_sdk.trace
    def create_cache(self, contents: str, display_name: str, ttl: int = 3600) -> str | None:
        """
        Create a cache for the given content and display name. We will use the display name as the key.
        If the cache already exists, it will be updated with the new content.

        Args:
            content: The content to cache.
            display_name: The display name to be used as the key of the cache.
            ttl: The time to live (in seconds) for the cache. Defaults to 1 hour.
        Returns:
            Cache name as specified by Gemini.
        """
        client = self.get_client()

        # We cannot get the cache name from the display name, only from the generated name which we do not have betweeen sessions
        # So we must do an O(n) search to find the cache by display name
        caches = client.caches.list()
        for cache in caches:
            if cache.display_name == display_name and cache.name:
                return cache.name

        cache = client.caches.create(
            model=self.model_name,
            config=CreateCachedContentConfig(
                display_name=display_name,
                contents=contents,
                ttl=f"{ttl}s",
            ),
        )
        return cache.name

    @sentry_sdk.trace
    def get_cache(self, display_name: str) -> str | None:
        client = self.get_client()

        # We cannot get the cache name from the display_name, only from the generated name which we do not have betweeen sessions
        # So we must do an O(n) search to find the cache by display name
        caches = client.caches.list()
        for cache in caches:
            if cache.display_name == display_name:
                return cache.name
        return None

    def copy(self, **kwargs):
        model = super().copy(**kwargs)
        model.local_regions_only = self.local_regions_only
        return model


LlmProvider = Union[OpenAiProvider, AnthropicProvider, GeminiProvider]


class LlmClient:
    @staticmethod
    def _resolve_parameter(
        explicit_value: Any,
        defaults: LlmProviderDefaults | None,
        param_name: str,
        fallback_default: Any = None,
    ) -> Any:
        """Helper to resolve a parameter using precedence: explicit > defaults > fallback."""
        if explicit_value is not None:
            return explicit_value
        if defaults and getattr(defaults, param_name, None) is not None:
            return getattr(defaults, param_name)
        return fallback_default

    @staticmethod
    def _resolve_parameters(
        *,
        defaults: LlmProviderDefaults | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        first_token_timeout: float | None = None,
        inactivity_timeout: float | None = None,
    ) -> ResolvedParameters:
        """
        Resolve parameters using precedence: function params > model defaults > provider defaults.
        Returns a ResolvedParameters Pydantic model with resolved parameter values.
        """
        return ResolvedParameters(
            temperature=LlmClient._resolve_parameter(temperature, defaults, "temperature"),
            max_tokens=LlmClient._resolve_parameter(max_tokens, defaults, "max_tokens"),
            seed=LlmClient._resolve_parameter(seed, defaults, "seed"),
            reasoning_effort=LlmClient._resolve_parameter(
                reasoning_effort, defaults, "reasoning_effort"
            ),
            timeout=LlmClient._resolve_parameter(timeout, defaults, "timeout"),
            first_token_timeout=LlmClient._resolve_parameter(
                first_token_timeout, defaults, "first_token_timeout", DEFAULT_FIRST_TOKEN_TIMEOUT
            ),
            inactivity_timeout=LlmClient._resolve_parameter(
                inactivity_timeout, defaults, "inactivity_timeout", DEFAULT_INACTIVITY_TIMEOUT
            ),
        )

    @staticmethod
    def _is_fallback_worthy_exception(exception: Exception, provider: LlmProvider) -> bool:
        """
        Determine if an exception warrants trying the next model in a fallback sequence.
        Currently targets 429 (rate limit) errors.
        """

        if isinstance(exception, LlmStreamTimeoutError):
            return True

        if provider.is_completion_exception_retryable(exception):
            # We fallback on all retryable exceptions
            return True

        if provider.provider_name == LlmProviderType.OPENAI:
            return isinstance(exception, openai.RateLimitError) or (
                isinstance(exception, openai.APIStatusError) and exception.status_code in [429]
            )
        elif provider.provider_name == LlmProviderType.ANTHROPIC:
            return (
                isinstance(exception, anthropic.RateLimitError)
                or (
                    isinstance(exception, anthropic.APIStatusError)
                    and exception.status_code in [429]
                )
                or any(
                    message in str(exception)
                    for message in [
                        "overloaded_error",
                        "Quota exceeded",
                    ]
                )
            )
        elif provider.provider_name == LlmProviderType.GEMINI:
            return (
                isinstance(exception, ClientError)
                and exception.code in [429]
                or any(
                    message in str(exception)
                    for message in [
                        "Resource exhausted. Please try again later.",
                        "429 RESOURCE_EXHAUSTED",
                    ]
                )
            )

    def _execute_with_fallback(
        self,
        models: list[LlmProvider],
        operation_name: str,
        operation_func: Callable[[LlmProvider], Any],
    ) -> Any:
        """
        Execute an operation with fallback through a list of models.
        Tries each model in sequence if the previous one fails with a fallback-worthy exception.
        For each model, tries all regions in its region preference before moving to the next model.
        """
        for i, base_model in enumerate(models):
            regions_to_try = self._get_regions_to_try(base_model)

            if (
                not regions_to_try or (len(regions_to_try) == 1 and regions_to_try[0] is None)
            ) and base_model.provider_name != LlmProviderType.OPENAI:
                logger.error(f"No regions to run for model {base_model.model_name}.")

            # Try each region for this model
            for j, region in enumerate(regions_to_try):
                # Create a copy of the model with the specific region
                model_to_use = base_model.copy(region=region)

                try:
                    self._set_fallback_sentry_tags(model_to_use, i + 1, j + 1, len(models), region)

                    return operation_func(model_to_use)
                except Exception as e:
                    if self._is_fallback_worthy_exception(e, model_to_use):
                        is_last_region = j == len(regions_to_try) - 1
                        is_last_model = i == len(models) - 1

                        if not is_last_region:
                            logger.error(
                                f"{operation_name} failed with {model_to_use.provider_name} model '{model_to_use.model_name}' region '{region}' "
                                f"due to {type(e).__name__}: {str(e)}. "
                                f"Trying next region ({j + 2}/{len(regions_to_try)}): {regions_to_try[j + 1]}"
                            )
                            continue
                        elif not is_last_model:
                            logger.error(
                                f"{operation_name} failed with {model_to_use.provider_name} model '{model_to_use.model_name}' (all regions tried) "
                                f"due to {type(e).__name__}: {str(e)}. "
                                f"Trying next model ({i + 2}/{len(models)}): {models[i + 1].provider_name} '{models[i + 1].model_name}'"
                            )
                            break  # Break out of region loop to try next model
                        else:
                            # Last model and last region - fail
                            logger.error(
                                f"{operation_name} failed with all {len(models)} models and all regions. "
                                f"Last attempt with {model_to_use.provider_name} model '{model_to_use.model_name}' region '{region}' "
                                f"failed with {type(e).__name__}: {str(e)}"
                            )
                            raise e
                    else:
                        # Non-fallback-worthy exception, fail immediately
                        logger.error(
                            f"{operation_name} failed with {model_to_use.provider_name} model '{model_to_use.model_name}' region '{region}' "
                            f"due to non-retryable {type(e).__name__}: {str(e)}"
                        )
                        raise e

        raise ValueError("No models provided for fallback execution")

    def _get_regions_to_try(self, model: LlmProvider) -> list[str | None]:
        """Get the list of regions to try for a given model."""
        if model.region is not None:
            return [model.region]

        region_preference = model.get_region_preference()
        if region_preference is not None:
            # Cast to list[str | None] to handle type compatibility
            return list(region_preference)
        else:
            return [None]

    def _set_fallback_sentry_tags(
        self,
        model: LlmProvider,
        model_attempt: int,
        region_attempt: int,
        total_models: int,
        current_region: str | None,
    ) -> None:
        """Set Sentry tags for fallback tracking."""
        sentry_sdk.set_tag("llm_provider", model.provider_name)
        sentry_sdk.set_tag("llm_model_attempt", model_attempt)
        sentry_sdk.set_tag("llm_model_region_attempt", region_attempt)
        sentry_sdk.set_tag("llm_total_models", total_models)
        sentry_sdk.set_tag("llm_current_region", current_region)

    @observe(name="Generate Text")
    @sentry_sdk.trace
    def generate_text(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        model: LlmProvider | None = None,
        models: list[LlmProvider] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool | ClaudeTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        run_name: str | None = None,
        timeout: float | None = None,
        predicted_output: str | None = None,
        reasoning_effort: str | None = None,
    ) -> LlmGenerateTextResponse:
        # Validate input parameters
        if models is not None and model is not None:
            raise ValueError(
                "Cannot specify both 'model' and 'models' parameters. Use 'models' for fallback functionality or 'model' for single model usage."
            )

        if models is None and model is None:
            raise ValueError("Must specify either 'model' or 'models' parameter.")

        # Convert single model to list for unified processing
        models_to_try = models if models is not None else ([model] if model is not None else [])

        if not models_to_try:
            raise ValueError("At least one model must be provided.")

        def _generate_text_operation(model_to_use: LlmProvider) -> LlmGenerateTextResponse:
            if run_name:
                langfuse_context.update_current_observation(name=run_name + " - Generate Text")

            resolved = self._resolve_parameters(
                defaults=model_to_use.defaults,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )

            messages_cleaned = LlmClient.clean_message_content(messages if messages else [])
            if not tools:
                messages_cleaned = LlmClient.clean_tool_call_assistant_messages(messages_cleaned)

            if model_to_use.provider_name == LlmProviderType.OPENAI:
                openai_model = cast(OpenAiProvider, model_to_use)

                if tools and any(isinstance(tool, ClaudeTool) for tool in tools):
                    raise ValueError("Claude tools are not supported for OpenAI")

                function_tools = (
                    cast(
                        list[FunctionTool],
                        [tool for tool in tools if isinstance(tool, FunctionTool)],
                    )
                    if tools
                    else None
                )

                return openai_model.generate_text(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    seed=resolved.seed,
                    tools=function_tools,
                    timeout=resolved.timeout,
                    predicted_output=predicted_output,
                    reasoning_effort=resolved.reasoning_effort,
                )
            elif model_to_use.provider_name == LlmProviderType.ANTHROPIC:
                anthropic_model = cast(AnthropicProvider, model_to_use)
                return anthropic_model.generate_text(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    tools=tools,
                    timeout=resolved.timeout,
                    reasoning_effort=resolved.reasoning_effort,
                )
            elif model_to_use.provider_name == LlmProviderType.GEMINI:
                gemini_model = cast(GeminiProvider, model_to_use)

                if tools and any(isinstance(tool, ClaudeTool) for tool in tools):
                    raise ValueError("Claude tools are not supported for Gemini")

                function_tools = (
                    cast(
                        list[FunctionTool],
                        [tool for tool in tools if isinstance(tool, FunctionTool)],
                    )
                    if tools
                    else None
                )

                return gemini_model.generate_text(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    seed=resolved.seed,
                    tools=function_tools,
                )
            else:
                raise ValueError(f"Invalid provider: {model_to_use.provider_name}")

        try:
            return self._execute_with_fallback(
                models=models_to_try,
                operation_name="Text generation",
                operation_func=_generate_text_operation,
            )
        except Exception as e:
            logger.exception(f"Text generation failed with all provided models: {e}")
            raise e

    @observe(name="Generate Structured")
    @sentry_sdk.trace
    def generate_structured(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        model: LlmProvider | None = None,
        models: list[LlmProvider] | None = None,
        system_prompt: str | None = None,
        response_format: Type[StructuredOutputType],
        tools: list[FunctionTool | ClaudeTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        run_name: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        cache_name: str | None = None,
        thinking_budget: int | None = None,
    ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
        # Validate input parameters
        if models is not None and model is not None:
            raise ValueError(
                "Cannot specify both 'model' and 'models' parameters. Use 'models' for fallback functionality or 'model' for single model usage."
            )

        if models is None and model is None:
            raise ValueError("Must specify either 'model' or 'models' parameter.")

        # Convert single model to list for unified processing
        models_to_try = models if models is not None else ([model] if model is not None else [])

        if not models_to_try:
            raise ValueError("At least one model must be provided.")

        def _generate_structured_operation(
            model_to_use: LlmProvider,
        ) -> LlmGenerateStructuredResponse[StructuredOutputType]:
            if run_name:
                langfuse_context.update_current_observation(
                    name=run_name + " - Generate Structured"
                )

            resolved = self._resolve_parameters(
                defaults=model_to_use.defaults,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )

            messages_cleaned = LlmClient.clean_message_content(messages if messages else [])

            if model_to_use.provider_name == LlmProviderType.OPENAI:
                openai_model = cast(OpenAiProvider, model_to_use)

                if tools and any(isinstance(tool, ClaudeTool) for tool in tools):
                    raise ValueError("Claude tools are not supported for OpenAI")

                messages_cleaned = LlmClient.clean_tool_call_assistant_messages(messages_cleaned)
                function_tools = (
                    cast(
                        list[FunctionTool],
                        [tool for tool in tools if isinstance(tool, FunctionTool)],
                    )
                    if tools
                    else None
                )

                return openai_model.generate_structured(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    seed=resolved.seed,
                    tools=function_tools,
                    timeout=resolved.timeout,
                    reasoning_effort=resolved.reasoning_effort,
                )
            elif model_to_use.provider_name == LlmProviderType.ANTHROPIC:
                raise ValueError("Structured generation is not supported for Anthropic provider")
            elif model_to_use.provider_name == LlmProviderType.GEMINI:
                gemini_model = cast(GeminiProvider, model_to_use)

                if tools and any(isinstance(tool, ClaudeTool) for tool in tools):
                    raise ValueError("Claude tools are not supported for Gemini")

                function_tools = (
                    cast(
                        list[FunctionTool],
                        [tool for tool in tools if isinstance(tool, FunctionTool)],
                    )
                    if tools
                    else None
                )

                return gemini_model.generate_structured(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    seed=resolved.seed,
                    tools=function_tools,
                    cache_name=cache_name,
                    thinking_budget=thinking_budget,
                )
            else:
                raise ValueError(f"Invalid provider: {model_to_use.provider_name}")

        try:
            return self._execute_with_fallback(
                models=models_to_try,
                operation_name="Structured generation",
                operation_func=_generate_structured_operation,
            )
        except Exception as e:
            logger.exception(f"Structured generation failed with all provided models: {e}")
            raise e

    @observe(name="Generate Text Stream")
    @sentry_sdk.trace
    def generate_text_stream(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        model: LlmProvider | None = None,
        models: list[LlmProvider] | None = None,
        system_prompt: str | None = None,
        tools: list[FunctionTool | ClaudeTool] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        run_name: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        first_token_timeout: float | None = None,
        inactivity_timeout: float | None = None,
    ) -> Iterator[Tuple[str, str] | ToolCall | Usage | LlmProvider | StreamResetException]:
        # Validate input parameters
        if models is not None and model is not None:
            raise ValueError(
                "Cannot specify both 'model' and 'models' parameters. Use 'models' for fallback functionality or 'model' for single model usage."
            )

        if models is None and model is None:
            raise ValueError("Must specify either 'model' or 'models' parameter.")

        # Convert single model to list for unified processing
        models_to_try = models if models is not None else ([model] if model is not None else [])

        if not models_to_try:
            raise ValueError("At least one model must be provided.")

        def _generate_text_stream_operation(
            model_to_use: LlmProvider,
        ) -> Iterator[Tuple[str, str] | ToolCall | Usage | StreamResetException]:
            if run_name:
                langfuse_context.update_current_observation(
                    name=run_name + " - Generate Text Stream"
                )

            resolved = self._resolve_parameters(
                defaults=model_to_use.defaults,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                first_token_timeout=first_token_timeout,
                inactivity_timeout=inactivity_timeout,
            )

            messages_cleaned = LlmClient.clean_message_content(messages if messages else [])
            if not tools:
                messages_cleaned = LlmClient.clean_tool_call_assistant_messages(messages_cleaned)

            # Get the appropriate stream generator based on provider
            if model_to_use.provider_name == LlmProviderType.OPENAI:
                openai_model = cast(OpenAiProvider, model_to_use)

                if tools and any(isinstance(tool, ClaudeTool) for tool in tools):
                    raise ValueError("Claude tools are not supported for OpenAI")

                function_tools = (
                    cast(
                        list[FunctionTool],
                        [tool for tool in tools if isinstance(tool, FunctionTool)],
                    )
                    if tools
                    else None
                )

                return openai_model.generate_text_stream(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    seed=resolved.seed,
                    tools=function_tools,
                    timeout=resolved.timeout,
                    reasoning_effort=resolved.reasoning_effort,
                    first_token_timeout=resolved.first_token_timeout,
                    inactivity_timeout=resolved.inactivity_timeout,
                )
            elif model_to_use.provider_name == LlmProviderType.ANTHROPIC:
                anthropic_model = cast(AnthropicProvider, model_to_use)
                return anthropic_model.generate_text_stream(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    tools=tools,
                    timeout=resolved.timeout,
                    reasoning_effort=resolved.reasoning_effort,
                    first_token_timeout=resolved.first_token_timeout,
                    inactivity_timeout=resolved.inactivity_timeout,
                )
            elif model_to_use.provider_name == LlmProviderType.GEMINI:
                gemini_model = cast(GeminiProvider, model_to_use)

                if tools and any(isinstance(tool, ClaudeTool) for tool in tools):
                    raise ValueError("Claude tools are not supported for Gemini")

                function_tools = (
                    cast(
                        list[FunctionTool],
                        [tool for tool in tools if isinstance(tool, FunctionTool)],
                    )
                    if tools
                    else None
                )

                return gemini_model.generate_text_stream(
                    max_tokens=resolved.max_tokens,
                    messages=messages_cleaned,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=resolved.temperature,
                    seed=resolved.seed,
                    tools=function_tools,
                    first_token_timeout=resolved.first_token_timeout,
                    inactivity_timeout=resolved.inactivity_timeout,
                )
            else:
                raise ValueError(f"Invalid provider: {model_to_use.provider_name}")

        try:
            # For streaming, we need to handle fallback during iteration, not just during generator creation
            for i, base_model in enumerate(models_to_try):
                regions_to_try = self._get_regions_to_try(base_model)

                if (
                    not regions_to_try or (len(regions_to_try) == 1 and regions_to_try[0] is None)
                ) and base_model.provider_name != LlmProviderType.OPENAI:
                    logger.warning(f"No regions to run for model {base_model.model_name}.")

                # Try each region for this model
                for j, region in enumerate(regions_to_try):
                    # Create a copy of the model with the specific region
                    model_to_use = base_model.copy(region=region)

                    try:
                        self._set_fallback_sentry_tags(
                            model_to_use, i + 1, j + 1, len(models_to_try), region
                        )

                        # Get the generator for this model
                        generator = _generate_text_stream_operation(model_to_use)

                        # Try to consume the generator - exceptions will be raised here
                        for item in generator:
                            yield item

                        yield model_to_use

                        # If we get here, streaming was successful, so break out of all fallback loops
                        return

                    except Exception as e:
                        if self._is_fallback_worthy_exception(e, model_to_use):
                            is_last_region = j == len(regions_to_try) - 1
                            is_last_model = i == len(models_to_try) - 1

                            if not is_last_region:
                                # Try next region for this model
                                logger.warning(
                                    f"Text stream generation failed with {model_to_use.provider_name} model '{model_to_use.model_name}' region '{region}' "
                                    f"due to {type(e).__name__}: {str(e)}. "
                                    f"Trying next region ({j + 2}/{len(regions_to_try)}): {regions_to_try[j + 1]}"
                                )

                                yield StreamResetException()

                                continue
                            elif not is_last_model:
                                # Try next model
                                logger.warning(
                                    f"Text stream generation failed with {model_to_use.provider_name} model '{model_to_use.model_name}' (all regions tried) "
                                    f"due to {type(e).__name__}: {str(e)}. "
                                    f"Trying next model ({i + 2}/{len(models_to_try)}): {models_to_try[i + 1].provider_name} '{models_to_try[i + 1].model_name}'"
                                )

                                yield StreamResetException()

                                break  # Break out of region loop to try next model
                            else:
                                # Last model and last region - fail
                                logger.error(
                                    f"Text stream generation failed with all {len(models_to_try)} models and all regions. "
                                    f"Last attempt with {model_to_use.provider_name} model '{model_to_use.model_name}' region '{region}' "
                                    f"failed with {type(e).__name__}: {str(e)}"
                                )
                                raise e
                        else:
                            # Non-fallback-worthy exception, fail immediately
                            logger.error(
                                f"Text stream generation failed with {model_to_use.provider_name} model '{model_to_use.model_name}' region '{region}' "
                                f"due to non-retryable {type(e).__name__}: {str(e)}"
                            )
                            raise e

        except Exception as e:
            logger.exception(f"Text stream generation failed with all provided models: {e}")
            raise e

    @observe(name="Generate Text from Web Search")
    @sentry_sdk.trace
    def generate_text_from_web_search(
        self,
        *,
        prompt: str,
        model: LlmProvider | None = None,
        models: list[LlmProvider] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        run_name: str | None = None,
    ) -> str:
        # Validate input parameters
        if models is not None and model is not None:
            raise ValueError(
                "Cannot specify both 'model' and 'models' parameters. Use 'models' for fallback functionality or 'model' for single model usage."
            )

        if models is None and model is None:
            raise ValueError("Must specify either 'model' or 'models' parameter.")

        # Convert single model to list for unified processing
        models_to_try = models if models is not None else ([model] if model is not None else [])

        if not models_to_try:
            raise ValueError("At least one model must be provided.")

        def _generate_text_from_web_search_operation(model_to_use: LlmProvider) -> str:
            if run_name:
                langfuse_context.update_current_observation(name=run_name + " - Generate Text")

            resolved = self._resolve_parameters(
                defaults=model_to_use.defaults,
                temperature=temperature,
                seed=seed,
            )

            if model_to_use.provider_name == LlmProviderType.GEMINI:
                gemini_model = cast(GeminiProvider, model_to_use)
                return gemini_model.search_the_web(
                    prompt, temperature=resolved.temperature, seed=resolved.seed
                )
            else:
                raise ValueError(
                    f"Web search is only supported for Gemini provider, got: {model_to_use.provider_name}"
                )

        try:
            return self._execute_with_fallback(
                models=models_to_try,
                operation_name="Web search text generation",
                operation_func=_generate_text_from_web_search_operation,
            )
        except Exception as e:
            logger.exception(f"Web search text generation failed with all provided models: {e}")
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
                new_messages.append(
                    Message(
                        role="user",
                        content=(
                            message.content
                            if message.content and message.content.strip()
                            else "[empty result]"
                        ),
                        tool_calls=[],
                    )
                )
            elif message.role == "tool_use":
                new_messages.append(
                    Message(role="assistant", content=message.content, tool_calls=[])
                )
            else:
                new_messages.append(message)
        return new_messages

    @staticmethod
    def clean_assistant_messages(messages: list[Message]) -> list[Message]:
        new_messages = []
        for message in messages:
            if message.role == "assistant" or message.role == "tool_use":
                message.content = "."
                new_messages.append(message)
            else:
                new_messages.append(message)
        return new_messages

    @staticmethod
    def clean_message_content(messages: list[Message]) -> list[Message]:
        new_messages = []
        for message in messages:
            if not message.content:
                message.content = "."
            new_messages.append(message)
        return new_messages

    def construct_message_from_stream(
        self,
        content_chunks: list[str],
        tool_calls: list[ToolCall],
        model: LlmProvider,
        thinking_content_chunks: list[str] = [],
        thinking_signature: str | None = None,
    ) -> Message:
        if model.provider_name == LlmProviderType.OPENAI:
            openai_model = cast(OpenAiProvider, model)
            return openai_model.construct_message_from_stream(content_chunks, tool_calls)
        elif model.provider_name == LlmProviderType.ANTHROPIC:
            anthropic_model = cast(AnthropicProvider, model)
            return anthropic_model.construct_message_from_stream(
                content_chunks, tool_calls, thinking_content_chunks, thinking_signature
            )
        elif model.provider_name == LlmProviderType.GEMINI:
            gemini_model = cast(GeminiProvider, model)
            return gemini_model.construct_message_from_stream(content_chunks, tool_calls)
        else:
            raise ValueError(f"Invalid provider: {model.provider_name}")

    @sentry_sdk.trace
    def create_cache(
        self, contents: str, display_name: str, model: LlmProvider, ttl: int = 3600
    ) -> str:
        if model.provider_name == LlmProviderType.GEMINI:

            def _create_cache(model_to_use: LlmProvider):
                gemini_model = cast(GeminiProvider, model_to_use)
                gemini_model.local_regions_only = (
                    True  # Local region is required for caching to work
                )

                cache_name = gemini_model.create_cache(contents, display_name, ttl)
                if not cache_name:
                    raise ValueError("Failed to create cache")
                return cache_name

            return self._execute_with_fallback(
                models=[model],
                operation_name="Create cache",
                operation_func=_create_cache,
            )
        else:
            raise ValueError("Manual cache creation is only supported for Gemini.")

    @sentry_sdk.trace
    def get_cache(self, display_name: str, model: LlmProvider) -> str | None:
        if model.provider_name == LlmProviderType.GEMINI:

            def _create_cache(model_to_use: LlmProvider):
                gemini_model = cast(GeminiProvider, model_to_use)
                gemini_model.local_regions_only = (
                    True  # Local region is required for caching to work
                )
                return gemini_model.get_cache(display_name)

            return self._execute_with_fallback(
                models=[model],
                operation_name="Get cache",
                operation_func=_create_cache,
            )
        else:
            raise ValueError("Manual cache retrieval is only supported for Gemini.")


@module.provider
def provide_llm_client() -> LlmClient:
    return LlmClient()
