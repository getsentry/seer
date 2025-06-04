import json
import time

import pytest
from pydantic import BaseModel

from seer.automation.agent.client import (
    AnthropicProvider,
    BaseLlmProvider,
    GeminiProvider,
    LlmClient,
    OpenAiProvider,
    _iterate_with_timeouts,
)
from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmGenerateTextResponse,
    LlmProviderType,
    LlmRefusalError,
    LlmStreamFirstTokenTimeoutError,
    LlmStreamInactivityTimeoutError,
    Message,
    ToolCall,
    Usage,
)
from seer.automation.agent.tools import FunctionTool
from seer.configuration import AppConfig, provide_test_defaults
from seer.dependency_injection import Module


def _are_tool_calls_equal(tool_calls1: list[ToolCall], tool_calls2: list[ToolCall]) -> bool:
    for tool_call1, tool_call2 in zip(tool_calls1, tool_calls2, strict=True):
        tool_call1.id = None
        tool_call2.id = None
        if tool_call1 != tool_call2:
            return False
    return True


@pytest.mark.vcr()
def test_openai_generate_text():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    response = llm_client.generate_text(
        prompt="Say hello",
        model=model,
        seed=42,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Hello! How can I assist you today?"
    assert response.message.role == "assistant"
    assert response.metadata.model == "gpt-3.5-turbo"
    assert response.metadata.provider_name == LlmProviderType.OPENAI
    assert response.metadata.usage == Usage(completion_tokens=9, prompt_tokens=9, total_tokens=18)


@pytest.mark.vcr()
def test_openai_generate_text_with_models_list():
    """Test generate_text with models list (fallback)"""
    llm_client = LlmClient()
    models = [
        OpenAiProvider.model("gpt-3.5-turbo"),
        OpenAiProvider.model("gpt-4o-mini"),
    ]

    response = llm_client.generate_text(
        prompt="Say hello",
        models=models,
        seed=42,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Hello! How can I assist you today?"
    assert response.message.role == "assistant"
    assert response.metadata.model == "gpt-3.5-turbo"
    assert response.metadata.provider_name == LlmProviderType.OPENAI
    assert response.metadata.usage == Usage(completion_tokens=9, prompt_tokens=9, total_tokens=18)


@pytest.mark.vcr()
def test_anthropic_generate_text():
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    response = llm_client.generate_text(
        prompt="Say hello",
        model=model,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Hello! How can I assist you today?"
    assert response.message.role == "assistant"
    assert response.metadata.model == "claude-3-5-sonnet@20240620"
    assert response.metadata.provider_name == LlmProviderType.ANTHROPIC
    assert response.metadata.usage == Usage(completion_tokens=12, prompt_tokens=9, total_tokens=21)


@pytest.mark.vcr()
def test_anthropic_generate_text_with_models_list():
    """Test generate_text with Anthropic models list"""
    llm_client = LlmClient()
    models = [
        AnthropicProvider.model("claude-3-5-sonnet@20240620"),
        AnthropicProvider.model("claude-3-haiku@20240307"),
    ]

    response = llm_client.generate_text(
        prompt="Say hello",
        models=models,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Hello! How can I assist you today?"
    assert response.message.role == "assistant"
    assert response.metadata.model == "claude-3-5-sonnet@20240620"
    assert response.metadata.provider_name == LlmProviderType.ANTHROPIC
    assert response.metadata.usage == Usage(completion_tokens=12, prompt_tokens=9, total_tokens=21)


@pytest.mark.vcr()
def test_openai_generate_text_with_tools():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[
                {
                    "name": "x",
                    "type": "string",
                    "description": 'The string "Hello"',
                },
            ],
            fn=lambda x: x,
        )
    ]

    response = llm_client.generate_text(
        prompt="Invoke test_function please!",
        model=model,
        tools=tools,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content is None
    assert response.message.role == "assistant"
    assert _are_tool_calls_equal(
        response.message.tool_calls, [ToolCall(function="test_function", args='{"x":"Hello"}')]
    )


@pytest.mark.vcr()
def test_openai_generate_text_with_tools_models_list():
    """Test generate_text with tools using models list"""
    llm_client = LlmClient()
    models = [OpenAiProvider.model("gpt-3.5-turbo")]

    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[
                {
                    "name": "x",
                    "type": "string",
                    "description": 'The string "Hello"',
                },
            ],
            fn=lambda x: x,
        )
    ]

    response = llm_client.generate_text(
        prompt="Invoke test_function please!",
        models=models,
        tools=tools,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content is None
    assert response.message.role == "assistant"
    assert _are_tool_calls_equal(
        response.message.tool_calls, [ToolCall(function="test_function", args='{"x":"Hello"}')]
    )


@pytest.mark.vcr()
def test_anthropic_generate_text_with_tools():
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[
                {
                    "name": "x",
                    "type": "string",
                },
            ],
            fn=lambda x: x,
        )
    ]

    response = llm_client.generate_text(
        prompt="Please invoke test_function",
        model=model,
        tools=tools,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content is not None
    assert response.message.role == "tool_use"
    assert _are_tool_calls_equal(
        response.message.tool_calls, [ToolCall(function="test_function", args="{}")]
    )


@pytest.mark.vcr()
def test_anthropic_generate_text_with_tools_models_list():
    """Test generate_text with Anthropic tools using models list"""
    llm_client = LlmClient()
    models = [AnthropicProvider.model("claude-3-5-sonnet@20240620")]

    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[
                {
                    "name": "x",
                    "type": "string",
                },
            ],
            fn=lambda x: x,
        )
    ]

    response = llm_client.generate_text(
        prompt="Please invoke test_function",
        models=models,
        tools=tools,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content is not None
    assert response.message.role == "tool_use"
    assert _are_tool_calls_equal(
        response.message.tool_calls, [ToolCall(function="test_function", args="{}")]
    )


@pytest.mark.vcr()
def test_openai_generate_structured():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-4o-mini-2024-07-18")

    class TestStructure(BaseModel):
        name: str
        age: int

    response = llm_client.generate_structured(
        prompt="Generate a person",
        model=model,
        response_format=TestStructure,
        seed=42,
    )

    assert isinstance(response, LlmGenerateStructuredResponse)
    assert response.parsed == TestStructure(name="Alice Johnson", age=29)
    assert response.metadata.model == "gpt-4o-mini-2024-07-18"
    assert response.metadata.provider_name == LlmProviderType.OPENAI


@pytest.mark.vcr()
def test_openai_generate_structured_with_models_list():
    """Test generate_structured with models list"""
    llm_client = LlmClient()
    models = [OpenAiProvider.model("gpt-4o-mini-2024-07-18")]

    class TestStructure(BaseModel):
        name: str
        age: int

    response = llm_client.generate_structured(
        prompt="Generate a person named Alice Johnson with age 29",
        models=models,
        response_format=TestStructure,
        seed=42,
    )

    assert isinstance(response, LlmGenerateStructuredResponse)
    assert response.parsed == TestStructure(name="Alice Johnson", age=29)
    assert response.metadata.model == "gpt-4o-mini-2024-07-18"
    assert response.metadata.provider_name == LlmProviderType.OPENAI


def test_client_validation_both_model_and_models():
    """Test that LlmClient validates model/models parameters properly"""
    llm_client = LlmClient()

    with pytest.raises(ValueError, match="Cannot specify both 'model' and 'models'"):
        llm_client.generate_text(
            prompt="test",
            model=OpenAiProvider.model("gpt-4"),
            models=[OpenAiProvider.model("gpt-3.5-turbo")],
        )


def test_client_validation_neither_model_nor_models():
    """Test that LlmClient validates model/models parameters properly"""
    llm_client = LlmClient()

    with pytest.raises(ValueError, match="Must specify either 'model' or 'models'"):
        llm_client.generate_text(prompt="test")


def test_client_validation_empty_models_list():
    """Test that LlmClient validates empty models list"""
    llm_client = LlmClient()

    with pytest.raises(ValueError, match="At least one model must be provided"):
        llm_client.generate_text(prompt="test", models=[])


def test_provider_model_with_custom_parameters():
    """Test provider .model() method with custom parameters"""
    # Test OpenAI provider with custom parameters
    openai_model = OpenAiProvider.model(
        "gpt-4",
        temperature=0.5,
        max_tokens=1000,
        seed=42,
        reasoning_effort="high",
    )
    assert openai_model.model_name == "gpt-4"
    assert openai_model.defaults.temperature == 0.5
    assert openai_model.defaults.max_tokens == 1000
    assert openai_model.defaults.seed == 42
    assert openai_model.defaults.reasoning_effort == "high"

    # Test Anthropic provider with custom parameters
    anthropic_model = AnthropicProvider.model(
        "claude-3-5-sonnet",
        region="us-east-1",
        temperature=0.7,
        max_tokens=2000,
        timeout=30.0,
    )
    assert anthropic_model.model_name == "claude-3-5-sonnet"
    assert anthropic_model.region == "us-east-1"
    assert anthropic_model.defaults.temperature == 0.7
    assert anthropic_model.defaults.max_tokens == 2000
    assert anthropic_model.defaults.timeout == 30.0

    # Test Gemini provider with custom parameters
    gemini_model = GeminiProvider.model(
        "gemini-pro",
        region="us-central1",
        temperature=0.3,
        max_tokens=1500,
        seed=123,
        local_regions_only=True,
    )
    assert gemini_model.model_name == "gemini-pro"
    assert gemini_model.region == "us-central1"
    assert gemini_model.defaults.temperature == 0.3
    assert gemini_model.defaults.max_tokens == 1500
    assert gemini_model.defaults.seed == 123
    assert gemini_model.local_regions_only is True


def test_openai_o_model_defaults():
    """Test that OpenAI O models get specific default configurations"""
    o1_model = OpenAiProvider.model("o1-mini")
    assert o1_model.defaults.temperature == 1.0
    # The first_token_timeout is in the base config but not transferred to merged defaults
    # This is expected behavior based on the current implementation

    o3_model = OpenAiProvider.model("o3-mini")
    assert o3_model.defaults.temperature == 1.0

    # Regular model should have regular defaults
    gpt4_model = OpenAiProvider.model("gpt-4")
    assert gpt4_model.defaults.temperature == 0.0  # Default from general config

    # Test that the base config for O models has the right values
    o1_config = OpenAiProvider.get_config("o1-mini")
    assert o1_config is not None
    assert o1_config.defaults.temperature == 1.0
    assert o1_config.defaults.first_token_timeout == 90.0

    # Test that regular models don't get O model defaults
    gpt4_config = OpenAiProvider.get_config("gpt-4")
    assert gpt4_config is not None
    assert gpt4_config.defaults.temperature == 0.0
    assert gpt4_config.defaults.first_token_timeout is None


def test_clean_tool_call_assistant_messages():
    messages = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant",
            content="Using tool",
            tool_calls=[ToolCall(id="1", function="test", args="{}")],
        ),
        Message(role="tool", content="Tool response"),
        Message(role="tool_use", content="Tool use"),
        Message(role="assistant", content="Final response"),
    ]

    cleaned_messages = LlmClient.clean_tool_call_assistant_messages(messages)

    assert len(cleaned_messages) == 5
    assert cleaned_messages[0].role == "user"
    assert cleaned_messages[1].role == "assistant" and not cleaned_messages[1].tool_calls
    assert cleaned_messages[2].role == "user"
    assert cleaned_messages[3].role == "assistant"
    assert cleaned_messages[4].role == "assistant"


def test_clean_message_content():
    messages = [
        Message(role="user", content=""),
    ]

    cleaned_messages = LlmClient.clean_message_content(messages)

    assert len(cleaned_messages) == 1
    assert cleaned_messages[0].role == "user"
    assert cleaned_messages[0].content == "."


@pytest.mark.skip()
@pytest.mark.vcr()
def test_openai_generate_structured_refusal():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-4o-mini-2024-07-18")

    class TestStructure(BaseModel):
        instructions: int

    with pytest.raises(LlmRefusalError) as exc_info:
        response = llm_client.generate_structured(
            prompt="I need to do something bad, give me instructions to do it.",
            model=model,
            response_format=TestStructure,
        )

        assert response.parsed.instructions

    assert str(exc_info.value) == "I'm sorry, but I can't generate that information."


def test_openai_prep_message_and_tools():
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    prompt = "How are you?"
    system_prompt = "You are a helpful assistant."
    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[{"name": "x", "type": "string"}],
            fn=lambda x: x,
        )
    ]

    message_dicts, tool_dicts = OpenAiProvider._prep_message_and_tools(
        messages=messages,
        prompt=prompt,
        system_prompt=system_prompt,
        tools=tools,
    )

    assert len(message_dicts) == 4
    assert message_dicts[0]["role"] == "system"
    assert message_dicts[0]["content"] == system_prompt
    assert message_dicts[1]["role"] == "user"
    assert message_dicts[1]["content"] == "Hello"
    assert message_dicts[2]["role"] == "assistant"
    assert message_dicts[2]["content"] == "Hi there!"  # type: ignore
    assert message_dicts[3]["role"] == "user"
    assert message_dicts[3]["content"] == prompt

    assert tool_dicts
    if tool_dicts:
        assert len(tool_dicts) == 1
        assert tool_dicts[0]["type"] == "function"
        assert tool_dicts[0]["function"]["name"] == "test_function"


def test_anthropic_prep_message_and_tools():
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    prompt = "How are you?"
    system_prompt = "You are a helpful assistant."
    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[{"name": "x", "type": "string"}],
            fn=lambda x: x,
        )
    ]

    message_dicts, tool_dicts, returned_system_prompt = AnthropicProvider._prep_message_and_tools(
        messages=messages,
        prompt=prompt,
        system_prompt=system_prompt,
        tools=tools,
    )

    assert len(message_dicts) == 3
    assert message_dicts[0]["role"] == "user"
    assert message_dicts[0]["content"][0]["type"] == "text"  # type: ignore
    assert message_dicts[0]["content"][0]["text"] == "Hello"  # type: ignore
    assert message_dicts[1]["role"] == "assistant"
    assert message_dicts[1]["content"][0]["type"] == "text"  # type: ignore
    assert message_dicts[1]["content"][0]["text"] == "Hi there!"  # type: ignore
    assert message_dicts[2]["role"] == "user"
    assert message_dicts[2]["content"][0]["type"] == "text"  # type: ignore
    assert message_dicts[2]["content"][0]["text"] == prompt  # type: ignore

    assert tool_dicts
    if tool_dicts:
        assert len(tool_dicts) == 1
        assert tool_dicts[0]["name"] == "test_function"
        assert "description" in tool_dicts[0]
        assert "input_schema" in tool_dicts[0]

    assert returned_system_prompt[0]["text"] == system_prompt


@pytest.mark.vcr()
def test_openai_generate_text_stream():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Say hello",
            model=model,
            seed=42,
        )
    )

    # Check that we got content chunks, usage, and the final model used
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]
    provider_items = [item for item in stream_items if hasattr(item, "provider_name")]

    assert len(content_chunks) > 0
    assert "".join(content_chunks) == "Hello! How can I assist you today?"
    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens == 9
    assert usage_items[0].prompt_tokens == 9
    assert usage_items[0].total_tokens == 18

    # Check that the final model used is yielded
    assert len(provider_items) == 1
    assert provider_items[0].model_name == "gpt-3.5-turbo"
    assert provider_items[0].provider_name == LlmProviderType.OPENAI


@pytest.mark.vcr()
def test_openai_generate_text_stream_with_models_list():
    """Test generate_text_stream with models list"""
    llm_client = LlmClient()
    models = [OpenAiProvider.model("gpt-3.5-turbo")]

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Say hello",
            models=models,
            seed=42,
        )
    )

    # Check that we got content chunks, usage, and the final model used
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]
    provider_items = [item for item in stream_items if hasattr(item, "provider_name")]

    assert len(content_chunks) > 0
    assert "".join(content_chunks) == "Hello! How can I assist you today?"
    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens == 9
    assert usage_items[0].prompt_tokens == 9
    assert usage_items[0].total_tokens == 18

    # Check that the final model used is yielded
    assert len(provider_items) == 1
    assert provider_items[0].model_name == "gpt-3.5-turbo"
    assert provider_items[0].provider_name == LlmProviderType.OPENAI


@pytest.mark.vcr()
def test_anthropic_generate_text_stream():
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Say hello",
            model=model,
        )
    )

    # Check that we got content chunks, usage, and the final model used
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]
    provider_items = [item for item in stream_items if hasattr(item, "provider_name")]

    assert len(content_chunks) > 0
    assert "".join(content_chunks) == "Hello! How can I assist you today?"
    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens > 0
    assert usage_items[0].prompt_tokens > 0
    assert (
        usage_items[0].total_tokens
        == usage_items[0].completion_tokens + usage_items[0].prompt_tokens
    )

    # Check that the final model used is yielded
    assert len(provider_items) == 1
    assert provider_items[0].model_name == "claude-3-5-sonnet@20240620"
    assert provider_items[0].provider_name == LlmProviderType.ANTHROPIC


@pytest.mark.vcr()
def test_openai_generate_text_stream_with_tools():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-4o-2024-08-06")

    tools = [
        FunctionTool(
            name="test_function",
            description="A test function that takes a string input",
            parameters=[
                {
                    "name": "x",
                    "type": "string",
                    "description": "The string to process",
                },
            ],
            fn=lambda x: x,
        )
    ]

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Call test_function with the input 'test data'",
            model=model,
            tools=tools,
            seed=42,
        )
    )

    # Check that we got tool calls, usage, and the final model used
    tool_calls = [item for item in stream_items if isinstance(item, ToolCall)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]
    provider_items = [item for item in stream_items if hasattr(item, "provider_name")]

    if tool_calls:
        assert len(tool_calls) >= 1
        tool_call = tool_calls[0]
        assert tool_call.function == "test_function"
        assert tool_call.id is not None
        args = json.loads(tool_call.args)
        assert "x" in args
        assert isinstance(args["x"], str)

    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens > 0
    assert usage_items[0].prompt_tokens > 0
    assert (
        usage_items[0].total_tokens
        == usage_items[0].completion_tokens + usage_items[0].prompt_tokens
    )

    # Check that the final model used is yielded
    assert len(provider_items) == 1
    assert provider_items[0].model_name == "gpt-4o-2024-08-06"
    assert provider_items[0].provider_name == LlmProviderType.OPENAI


@pytest.mark.vcr()
def test_anthropic_generate_text_stream_with_tools():
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[
                {
                    "name": "x",
                    "type": "string",
                },
            ],
            fn=lambda x: x,
        )
    ]

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Please invoke test_function",
            model=model,
            tools=tools,
        )
    )

    # Check that we got content chunks, tool calls, usage, and the final model used
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    tool_calls = [item for item in stream_items if isinstance(item, ToolCall)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]
    provider_items = [item for item in stream_items if hasattr(item, "provider_name")]

    assert len(content_chunks) > 0
    assert len(tool_calls) == 1
    assert tool_calls[0].function == "test_function"
    assert len(usage_items) == 1

    # Check that the final model used is yielded
    assert len(provider_items) == 1
    assert provider_items[0].model_name == "claude-3-5-sonnet@20240620"
    assert provider_items[0].provider_name == LlmProviderType.ANTHROPIC


def test_construct_message_from_stream_openai():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    content_chunks = ["Hello", " world", "!"]
    tool_calls = [ToolCall(id="123", function="test_function", args='{"x": "test"}')]

    message = llm_client.construct_message_from_stream(
        content_chunks=content_chunks,
        tool_calls=tool_calls,
        model=model,
    )

    assert message.role == "assistant"
    assert message.content == "Hello world!"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "123"
    assert message.tool_calls[0].function == "test_function"
    assert message.tool_calls[0].args == '{"x": "test"}'


def test_construct_message_from_stream_anthropic():
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    content_chunks = ["Hello", " world", "!"]
    tool_calls = [ToolCall(id="123", function="test_function", args='{"x": "test"}')]

    message = llm_client.construct_message_from_stream(
        content_chunks=content_chunks,
        tool_calls=tool_calls,
        model=model,
    )

    assert message.role == "tool_use"
    assert message.content == "Hello world!"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "123"
    assert message.tool_calls[0].function == "test_function"
    assert message.tool_calls[0].args == '{"x": "test"}'
    assert message.tool_call_id == "123"


def test_construct_message_from_stream_invalid_provider():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")
    model.provider_name = "invalid"  # type: ignore

    with pytest.raises(ValueError, match="Invalid provider: invalid"):
        llm_client.construct_message_from_stream(
            content_chunks=["test"],
            tool_calls=[],
            model=model,
        )


@pytest.mark.vcr()
def test_gemini_generate_text():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-001")

    response = llm_client.generate_text(
        prompt="Say hello",
        model=model,
        seed=42,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content is not None
    assert "hello" in response.message.content.lower()
    assert response.message.role == "assistant"
    assert response.metadata.model == "gemini-2.0-flash-001"
    assert response.metadata.provider_name == LlmProviderType.GEMINI
    assert response.metadata.usage == Usage(completion_tokens=11, prompt_tokens=2, total_tokens=13)


@pytest.mark.vcr()
def test_gemini_generate_text_with_models_list():
    """Test generate_text with Gemini models list"""
    llm_client = LlmClient()
    models = [GeminiProvider.model("gemini-2.0-flash-001")]

    response = llm_client.generate_text(
        prompt="Say hello",
        models=models,
        seed=42,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content is not None
    assert "hello" in response.message.content.lower()
    assert response.message.role == "assistant"
    assert response.metadata.model == "gemini-2.0-flash-001"
    assert response.metadata.provider_name == LlmProviderType.GEMINI
    assert response.metadata.usage == Usage(completion_tokens=11, prompt_tokens=2, total_tokens=13)


@pytest.mark.vcr()
def test_gemini_generate_text_with_tools():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-001")

    tools = [
        FunctionTool(
            name="submit",
            description="A test function",
            parameters=[
                {
                    "name": "complete",
                    "type": "boolean",
                },
            ],
            fn=lambda complete: complete,
        )
    ]

    response = llm_client.generate_text(
        prompt="Please write a haiku, then invoke 'submit' with complete = true",
        model=model,
        tools=tools,
        seed=42,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content is not None
    assert len(response.message.content) > 0
    assert response.message.role == "tool_use"
    assert response.message.tool_calls is not None
    assert response.message.tool_calls == [
        ToolCall(
            function="submit",
            args='{"complete": true}',
        ),
    ]


@pytest.mark.vcr()
def test_gemini_generate_structured():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-001")

    class TestStructure(BaseModel):
        name: str
        age: int

    response = llm_client.generate_structured(
        prompt="Generate a person named John Doe aged 30",
        model=model,
        response_format=TestStructure,
        seed=42,
    )

    assert isinstance(response, LlmGenerateStructuredResponse)
    assert response.parsed == TestStructure(name="John Doe", age=30)
    assert response.metadata.model == "gemini-2.0-flash-001"
    assert response.metadata.provider_name == LlmProviderType.GEMINI


@pytest.mark.vcr()
def test_gemini_generate_structured_with_models_list():
    """Test generate_structured with Gemini models list"""
    llm_client = LlmClient()
    models = [GeminiProvider.model("gemini-2.0-flash-001")]

    class TestStructure(BaseModel):
        name: str
        age: int

    response = llm_client.generate_structured(
        prompt="Generate a person named John Doe aged 30",
        models=models,
        response_format=TestStructure,
        seed=42,
    )

    assert isinstance(response, LlmGenerateStructuredResponse)
    assert response.parsed == TestStructure(name="John Doe", age=30)
    assert response.metadata.model == "gemini-2.0-flash-001"
    assert response.metadata.provider_name == LlmProviderType.GEMINI


def test_gemini_prep_message_and_tools():
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    prompt = "How are you?"
    system_prompt = "You are a helpful assistant."
    tools = [
        FunctionTool(
            name="test_function",
            description="A test function",
            parameters=[{"name": "x", "type": "string"}],
            fn=lambda x: x,
        )
    ]

    message_dicts, tool_dicts, returned_system_prompt = GeminiProvider._prep_message_and_tools(
        messages=messages,
        prompt=prompt,
        system_prompt=system_prompt,
        tools=tools,
    )

    assert len(message_dicts) == 3
    assert message_dicts[0].role == "user"
    assert message_dicts[0].parts is not None
    assert len(message_dicts[0].parts) > 0
    assert message_dicts[0].parts[0].text == "Hello"
    assert message_dicts[1].role == "model"
    assert message_dicts[1].parts is not None
    assert len(message_dicts[1].parts) > 0
    assert message_dicts[1].parts[0].text == "Hi there!"
    assert message_dicts[2].role == "user"
    assert message_dicts[2].parts is not None
    assert len(message_dicts[2].parts) > 0
    assert message_dicts[2].parts[0].text == prompt

    assert tool_dicts is not None
    assert len(tool_dicts) == 1
    if tool_dicts:
        assert tool_dicts[0].function_declarations
        if tool_dicts[0].function_declarations:
            assert len(tool_dicts[0].function_declarations) > 0
            assert tool_dicts[0].function_declarations[0].name == "test_function"
            assert tool_dicts[0].function_declarations[0].description == "A test function"
            assert tool_dicts[0].function_declarations[0].parameters
            if tool_dicts[0].function_declarations[0].parameters:
                assert tool_dicts[0].function_declarations[0].parameters.type == "OBJECT"
                assert "x" in tool_dicts[0].function_declarations[0].parameters.properties
                assert (
                    tool_dicts[0].function_declarations[0].parameters.properties["x"].type
                    == "STRING"
                )


@pytest.mark.vcr()
def test_gemini_generate_text_stream():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.5-flash-preview-04-17")

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Say hello",
            model=model,
            seed=42,
        )
    )

    # Check that we got content chunks, usage, and the final model used
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]
    provider_items = [item for item in stream_items if hasattr(item, "provider_name")]

    assert len(content_chunks) > 0
    assert "hello" in "".join(content_chunks).lower()
    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens > 0
    assert usage_items[0].prompt_tokens > 0
    assert (
        usage_items[0].total_tokens
        == usage_items[0].completion_tokens + usage_items[0].prompt_tokens
    )

    # Check that the final model used is yielded
    assert len(provider_items) == 1
    assert provider_items[0].model_name == "gemini-2.5-flash-preview-04-17"
    assert provider_items[0].provider_name == LlmProviderType.GEMINI


@pytest.mark.vcr()
def test_gemini_generate_text_stream_with_tools():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.5-flash-preview-04-17")

    tools = [
        FunctionTool(
            name="submit",
            description="A test function",
            parameters=[
                {
                    "name": "complete",
                    "type": "boolean",
                },
            ],
            fn=lambda complete: complete,
        )
    ]

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Please write a haiku, then invoke 'submit' with complete = true",
            model=model,
            tools=tools,
            seed=42,
        )
    )

    # Check that we got content chunks, tool calls, usage, and the final model used
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    tool_calls = [item for item in stream_items if isinstance(item, ToolCall)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]
    provider_items = [item for item in stream_items if hasattr(item, "provider_name")]

    assert len(content_chunks) > 0
    assert len(tool_calls) == 1
    assert tool_calls[0].function == "submit"
    assert tool_calls[0].args == '{"complete": true}'
    assert len(usage_items) == 1

    # Check that the final model used is yielded
    assert len(provider_items) == 1
    assert provider_items[0].model_name == "gemini-2.5-flash-preview-04-17"
    assert provider_items[0].provider_name == LlmProviderType.GEMINI


def test_construct_message_from_stream_gemini():
    model = GeminiProvider.model("gemini-2.0-flash-001")

    content_chunks = ["Hello", " world", "!"]
    tool_calls = [ToolCall(id="123", function="test_function", args='{"x": "test"}')]

    message = model.construct_message_from_stream(content_chunks, tool_calls)

    assert message.role == ("tool_use" if tool_calls else "assistant")
    assert message.content == "Hello world!"
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "123"
    assert message.tool_calls[0].function == "test_function"
    assert message.tool_calls[0].args == '{"x": "test"}'
    assert message.tool_call_id == "123"


@pytest.mark.vcr()
def test_gemini_generate_text_from_web_search():
    llm_client = LlmClient()
    model = GeminiProvider(model_name="gemini-2.0-flash-001")

    response = llm_client.generate_text_from_web_search(
        prompt="What year is it?",
        model=model,
        seed=42,
    )

    assert isinstance(response, str)
    assert "2025" in response


@pytest.mark.vcr()
def test_gemini_generate_text_from_web_search_with_models_list():
    """Test web search with models list"""
    llm_client = LlmClient()
    models = [GeminiProvider(model_name="gemini-2.0-flash-001")]

    response = llm_client.generate_text_from_web_search(
        prompt="What year is it?",
        models=models,
        seed=42,
    )

    assert isinstance(response, str)
    assert "2025" in response


@pytest.mark.vcr()
def test_gemini_create_cache():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-001")

    contents = "test" * 5000  # Min cache is 4096

    cache_name = llm_client.create_cache(
        contents=contents,
        display_name="test_cache",
        model=model,
        ttl=3600,
    )

    assert isinstance(cache_name, str)
    assert len(cache_name) > 0


def test_gemini_create_cache_invalid_provider():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    with pytest.raises(ValueError, match="Manual cache creation is only supported for Gemini"):
        llm_client.create_cache(
            contents="Test content",
            display_name="test_cache",
            model=model,
        )


@pytest.mark.vcr()
def test_gemini_create_cache_same_display_name():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-001")

    contents = "test" * 5000  # Min cache is 4096

    # Create cache with default TTL (3600)
    cache_name1 = llm_client.create_cache(
        contents=contents,
        display_name="test_cache_same_display_name",
        model=model,
    )

    # Create cache with custom TTL
    cache_name2 = llm_client.create_cache(
        contents=contents,
        display_name="test_cache_same_display_name",
        model=model,
        ttl=7200,
    )

    assert isinstance(cache_name1, str)
    assert isinstance(cache_name2, str)
    assert cache_name1 == cache_name2


@pytest.mark.vcr()
def test_gemini_get_cache():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-001")

    contents = "test" * 5000  # Min cache is 4096
    original_cache = llm_client.create_cache(
        contents=contents,
        display_name="test_cache_get",
        model=model,
    )

    retrieved_cache = llm_client.get_cache(display_name="test_cache_get", model=model)

    assert original_cache == retrieved_cache


def test_iterate_with_timeouts_success_and_timeouts():

    # No delays: should yield all items
    def gen_quick():
        yield "x"
        yield "y"

    assert list(
        _iterate_with_timeouts(gen_quick(), first_token_timeout=0.1, inactivity_timeout=0.1)
    ) == ["x", "y"]

    # Delay before first token: should timeout
    def gen_slow_first():
        time.sleep(0.2)
        yield "x"

    with pytest.raises(LlmStreamFirstTokenTimeoutError):
        list(
            _iterate_with_timeouts(
                gen_slow_first(), first_token_timeout=0.1, inactivity_timeout=0.1
            )
        )

    # Delay between tokens: should timeout on inactivity
    def gen_slow_inactivity():
        yield "x"
        time.sleep(0.2)
        yield "y"

    with pytest.raises(LlmStreamInactivityTimeoutError):
        list(
            _iterate_with_timeouts(
                gen_slow_inactivity(), first_token_timeout=0.1, inactivity_timeout=0.1
            )
        )


def test_iterate_with_timeouts_error_and_cleanup(monkeypatch):
    from seer.automation.agent.client import _iterate_with_timeouts

    called = False

    def on_cleanup():
        nonlocal called
        called = True

    def gen_error():
        raise RuntimeError("fail")
        yield

    with pytest.raises(RuntimeError):
        list(
            _iterate_with_timeouts(
                gen_error(), first_token_timeout=0.1, inactivity_timeout=0.1, on_cleanup=on_cleanup
            )
        )
    assert called


@pytest.mark.vcr()
def test_gemini_thinking_config():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.5-flash-preview-04-17")

    class TestStructure(BaseModel):
        name: str
        age: int

    # Test with thinking budget
    response_with_budget = llm_client.generate_structured(
        prompt="Generate a person named John Doe aged 30",
        model=model,
        response_format=TestStructure,
        thinking_budget=1024,
    )

    assert isinstance(response_with_budget, LlmGenerateStructuredResponse)
    assert response_with_budget.parsed == TestStructure(name="John Doe", age=30)
    assert response_with_budget.metadata.model == "gemini-2.5-flash-preview-04-17"
    assert response_with_budget.metadata.provider_name == LlmProviderType.GEMINI

    # Test without thinking budget
    response_without_budget = llm_client.generate_structured(
        prompt="Generate a person named Jane Doe aged 25",
        model=model,
        response_format=TestStructure,
        thinking_budget=0,
    )

    assert isinstance(response_without_budget, LlmGenerateStructuredResponse)
    assert response_without_budget.parsed == TestStructure(name="Jane Doe", age=25)
    assert response_without_budget.metadata.model == "gemini-2.5-flash-preview-04-17"
    assert response_without_budget.metadata.provider_name == LlmProviderType.GEMINI


def test_parameter_resolution():
    """Test parameter resolution logic"""
    llm_client = LlmClient()

    # Test _resolve_parameters method directly
    from seer.automation.agent.models import LlmProviderDefaults

    defaults = LlmProviderDefaults(temperature=0.5, max_tokens=1000, seed=42)

    resolved = llm_client._resolve_parameters(
        defaults=defaults,
        temperature=0.7,  # Override default
        max_tokens=None,  # Use default
        seed=None,  # Use default
        reasoning_effort="high",  # No default, use provided
    )

    assert resolved.temperature == 0.7  # Overridden
    assert resolved.max_tokens == 1000  # From defaults
    assert resolved.seed == 42  # From defaults
    assert resolved.reasoning_effort == "high"  # Provided
    assert resolved.first_token_timeout == 40.0  # Default constant
    assert resolved.inactivity_timeout == 20.0  # Default constant


def test_timeout_parameter_resolution():
    """Test timeout parameter resolution with custom values"""
    llm_client = LlmClient()

    from seer.automation.agent.models import LlmProviderDefaults

    defaults = LlmProviderDefaults(
        first_token_timeout=90.0,
        inactivity_timeout=30.0,
    )

    resolved = llm_client._resolve_parameters(
        defaults=defaults,
        first_token_timeout=120.0,  # Override default
        inactivity_timeout=None,  # Use default
    )

    assert resolved.first_token_timeout == 120.0  # Overridden
    assert resolved.inactivity_timeout == 30.0  # From defaults


def test_region_preference_functionality():
    """Test the new region preference system"""

    # Test US region preferences for Anthropic
    test_config = provide_test_defaults()
    test_config.SENTRY_REGION = "us"

    with Module().constant(AppConfig, test_config):
        anthropic_model = AnthropicProvider.model("claude-3-5-sonnet@20240620")
        region_prefs = anthropic_model.get_region_preference()
        assert region_prefs == ["us-east5", "global"]

    # Test DE region preferences for Anthropic
    test_config.SENTRY_REGION = "de"
    with Module().constant(AppConfig, test_config):
        region_prefs = anthropic_model.get_region_preference()
        assert region_prefs == ["europe-west4"]

    # Test Gemini region preferences
    test_config.SENTRY_REGION = "us"
    with Module().constant(AppConfig, test_config):
        gemini_model = GeminiProvider.model("gemini-2.0-flash-001")
        region_prefs = gemini_model.get_region_preference()
        assert region_prefs == ["us-central1", "us-east1", "global"]

    # Test preview model preferences (currently matches general config due to order)
    with Module().constant(AppConfig, test_config):
        gemini_preview = GeminiProvider.model("gemini-2.5-flash-preview-04-17")
        region_prefs = gemini_preview.get_region_preference()
        assert region_prefs == ["us-central1", "us-east1", "global"]  # Matches general config

    # Test local_regions_only functionality
    with Module().constant(AppConfig, test_config):
        gemini_local_only = GeminiProvider.model("gemini-2.0-flash-001", local_regions_only=True)
        region_prefs = gemini_local_only.get_region_preference()
        assert region_prefs == ["us-central1", "us-east1"]  # Global region filtered out


def test_region_preference_no_config():
    """Test region preference when no config is found"""

    # Test with model that doesn't match any config pattern
    class TestProvider(BaseLlmProvider):
        default_configs = []

    test_model = TestProvider(model_name="non-existent-model")

    test_config = provide_test_defaults()
    test_config.SENTRY_REGION = "us"

    with Module().constant(AppConfig, test_config):
        region_prefs = test_model.get_region_preference()
        assert region_prefs is None


def test_region_preference_unknown_sentry_region():
    """Test region preference with unknown SENTRY_REGION"""

    anthropic_model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    test_config = provide_test_defaults()
    test_config.SENTRY_REGION = "unknown"

    with Module().constant(AppConfig, test_config):
        region_prefs = anthropic_model.get_region_preference()
        assert region_prefs is None


def test_openai_provider_retry_logic():
    """Test that OpenAI provider applies retry logic to client methods"""
    from unittest.mock import Mock, patch

    provider = OpenAiProvider.model("gpt-3.5-turbo")

    # Mock the backoff_on_exception decorator
    with patch("seer.automation.agent.client.backoff_on_exception") as mock_backoff:
        mock_retrier = Mock()
        mock_backoff.return_value = mock_retrier

        provider.get_client()

        # Verify backoff_on_exception was called with the right parameters
        mock_backoff.assert_called_once_with(
            OpenAiProvider.is_completion_exception_retryable, max_tries=4
        )

        # Verify the retrier was applied to the correct client methods
        assert mock_retrier.call_count == 2  # Should be called twice for both methods


def test_anthropic_provider_retry_logic():
    """Test that Anthropic provider applies retry logic to client methods"""
    from unittest.mock import Mock, patch

    provider = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    # Mock the backoff_on_exception decorator
    with patch("seer.automation.agent.client.backoff_on_exception") as mock_backoff:
        mock_retrier = Mock()
        mock_backoff.return_value = mock_retrier

        provider.get_client()

        # Verify backoff_on_exception was called with the right parameters
        mock_backoff.assert_called_once_with(
            AnthropicProvider.is_completion_exception_retryable, max_tries=4
        )

        # Verify the retrier was applied to the correct client method
        assert mock_retrier.call_count == 1  # Should be called once for messages.create
