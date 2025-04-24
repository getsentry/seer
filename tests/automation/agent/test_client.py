import json
import time

import pytest
from pydantic import BaseModel

from seer.automation.agent.client import (
    AnthropicProvider,
    GeminiProvider,
    LlmClient,
    OpenAiProvider,
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


@pytest.mark.vcr()
def test_openai_generate_text():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    response = llm_client.generate_text(
        prompt="Say hello",
        model=model,
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
    assert response.message.tool_calls == [
        ToolCall(
            id="call_NMeKwqzR3emFbDdFNPGeI8E7", function="test_function", args='{"x": "Hello"}'
        ),
        ToolCall(
            id="call_dcOjoT13fP18vft0idWNqjRp", function="test_function", args='{"x": "World"}'
        ),
    ]


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
    assert response.message.tool_calls == [
        ToolCall(id="toolu_vrtx_01Y7rMxTGDBpGMDL1hwNY173", function="test_function", args="{}"),
    ]


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
    )

    assert isinstance(response, LlmGenerateStructuredResponse)
    assert response.parsed == TestStructure(name="Alice Johnson", age=28)
    assert response.metadata.model == "gpt-4o-mini-2024-07-18"
    assert response.metadata.provider_name == LlmProviderType.OPENAI


@pytest.mark.vcr()
def test_anthropic_generate_structured():
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    class TestStructure(BaseModel):
        name: str
        age: int

    with pytest.raises(NotImplementedError):
        llm_client.generate_structured(
            prompt="Generate a person",
            model=model,
            response_format=TestStructure,
        )


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
        )
    )

    # Check that we got content chunks and usage
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

    assert len(content_chunks) > 0
    assert "".join(content_chunks) == "Hello! How can I assist you today?"
    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens == 9
    assert usage_items[0].prompt_tokens == 9
    assert usage_items[0].total_tokens == 18


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

    # Check that we got content chunks and usage
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

    assert len(content_chunks) > 0
    assert "".join(content_chunks) == "Hello! How can I assist you today?"
    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens > 0
    assert usage_items[0].prompt_tokens > 0
    assert (
        usage_items[0].total_tokens
        == usage_items[0].completion_tokens + usage_items[0].prompt_tokens
    )


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
        )
    )

    # Check that we got tool calls and usage
    tool_calls = [item for item in stream_items if isinstance(item, ToolCall)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

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

    # Check that we got content chunks, tool calls and usage
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    tool_calls = [item for item in stream_items if isinstance(item, ToolCall)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

    assert len(content_chunks) > 0
    assert len(tool_calls) == 1
    assert tool_calls[0].function == "test_function"
    assert len(usage_items) == 1


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
    model = GeminiProvider.model("gemini-2.0-flash-001")

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Say hello",
            model=model,
        )
    )

    # Check that we got content chunks and usage
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

    assert len(content_chunks) > 0
    assert "hello" in "".join(content_chunks).lower()
    assert len(usage_items) == 1
    assert usage_items[0].completion_tokens > 0
    assert usage_items[0].prompt_tokens > 0
    assert (
        usage_items[0].total_tokens
        == usage_items[0].completion_tokens + usage_items[0].prompt_tokens
    )


@pytest.mark.vcr()
def test_gemini_generate_text_stream_with_tools():
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

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Please write a haiku, then invoke 'submit' with complete = true",
            model=model,
            tools=tools,
        )
    )

    # Check that we got content chunks, tool calls and usage
    content_chunks = [item[1] for item in stream_items if isinstance(item, tuple)]
    tool_calls = [item for item in stream_items if isinstance(item, ToolCall)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

    assert len(content_chunks) > 0
    assert len(tool_calls) == 1
    assert tool_calls[0].function == "submit"
    assert tool_calls[0].args == '{"complete": true}'
    assert len(usage_items) == 1


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
    )

    assert isinstance(response, str)
    assert "2025" in response


@pytest.mark.parametrize(
    "provider_class,model_name",
    [
        (OpenAiProvider, "gpt-3.5-turbo"),
        (AnthropicProvider, "claude-3-5-sonnet@20240620"),
        (GeminiProvider, "gemini-2.0-flash-001"),
    ],
)
def test_generate_text_stream_with_first_token_timeout(provider_class, model_name, monkeypatch):
    """Test that first token timeout is correctly applied."""
    llm_client = LlmClient()
    model = provider_class.model(model_name)

    def mock_stream_gen_with_first_token_delay(*args, **kwargs):
        # Simulate delay before yielding first token
        time.sleep(0.1)
        yield "First token"

        # Subsequent tokens come quickly
        for i in range(3):
            yield f"Token {i+2}"

        # Yield usage info at the end
        yield Usage(completion_tokens=5, prompt_tokens=5, total_tokens=10)

    monkeypatch.setattr(model, "generate_text_stream", mock_stream_gen_with_first_token_delay)

    # Should complete successfully with reasonable timeout values
    stream_items = list(
        llm_client.generate_text_stream(
            prompt="test",
            model=model,
            first_token_timeout=0.5,  # Long enough for our simulated delay
            inactivity_timeout=0.5,
        )
    )

    assert len(stream_items) == 5  # 4 tokens + usage info
    assert stream_items[0] == "First token"

    # Should timeout with too short first token timeout
    with pytest.raises(LlmStreamFirstTokenTimeoutError, match="time to first token timeout"):
        list(
            llm_client.generate_text_stream(
                prompt="test",
                model=model,
                first_token_timeout=0.05,  # Too short for our simulated delay
                inactivity_timeout=1.0,
            )
        )


@pytest.mark.parametrize(
    "provider_class,model_name",
    [
        (OpenAiProvider, "gpt-3.5-turbo"),
        (AnthropicProvider, "claude-3-5-sonnet@20240620"),
        (GeminiProvider, "gemini-2.0-flash-001"),
    ],
)
def test_generate_text_stream_with_inactivity_timeout(provider_class, model_name, monkeypatch):
    """Test that inactivity timeout is correctly applied after first token."""
    llm_client = LlmClient()
    model = provider_class.model(model_name)

    def mock_stream_gen_with_inactivity_delay(*args, **kwargs):
        # First token comes quickly
        yield "First token"

        # Simulate delay between first and second token
        time.sleep(0.2)
        yield "Second token"

        # More tokens come at normal speed
        for i in range(2):
            yield f"Token {i+3}"

        # Yield usage info at the end
        yield Usage(completion_tokens=5, prompt_tokens=5, total_tokens=10)

    monkeypatch.setattr(model, "generate_text_stream", mock_stream_gen_with_inactivity_delay)

    # Should complete successfully with reasonable timeout values
    stream_items = list(
        llm_client.generate_text_stream(
            prompt="test",
            model=model,
            first_token_timeout=1.0,
            inactivity_timeout=0.5,  # Long enough for our simulated delay
        )
    )

    assert len(stream_items) == 5  # 4 tokens + usage info
    assert stream_items[0] == "First token"
    assert stream_items[1] == "Second token"

    # Should timeout with too short inactivity timeout
    with pytest.raises(LlmStreamInactivityTimeoutError, match="inactivity timeout"):
        list(
            llm_client.generate_text_stream(
                prompt="test",
                model=model,
                first_token_timeout=1.0,
                inactivity_timeout=0.1,  # Too short for our simulated delay
            )
        )


def test_generate_text_stream_different_timeouts(monkeypatch):
    """Test that first token and inactivity timeouts are correctly differentiated."""
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    tokens_generated = []
    start_time = time.time()

    def mock_stream_gen(*args, **kwargs):
        nonlocal tokens_generated, start_time

        # Record when this generator is first called
        start_time = time.time()

        # First token takes 0.3 seconds
        time.sleep(0.3)
        # Record token only after it would be yielded (after timeout check)
        yield_token = "First token"
        yield yield_token
        tokens_generated.append(("first", time.time() - start_time))

        # Second token takes 0.2 seconds after first token
        time.sleep(0.2)
        yield_token = "Second token"
        yield yield_token
        tokens_generated.append(("second", time.time() - start_time))

        # Remaining tokens come quickly
        for i in range(3):
            time.sleep(0.05)
            yield_token = f"Token {i+3}"
            yield yield_token
            tokens_generated.append((f"token{i+3}", time.time() - start_time))

        # Yield usage info at the end
        yield Usage(completion_tokens=5, prompt_tokens=5, total_tokens=10)

    monkeypatch.setattr(model, "generate_text_stream", mock_stream_gen)

    # Should complete successfully with first token timeout > 0.3s and inactivity timeout > 0.2s
    tokens_generated = []
    stream_items = list(
        llm_client.generate_text_stream(
            prompt="test",
            model=model,
            first_token_timeout=0.5,
            inactivity_timeout=0.3,
        )
    )

    assert len(stream_items) == 6  # 5 tokens + usage info
    assert len(tokens_generated) == 5  # 5 tokens were generated
    assert tokens_generated[0][0] == "first"
    assert tokens_generated[1][0] == "second"

    # Should timeout on first token if first token timeout is too low
    tokens_generated = []
    with pytest.raises(LlmStreamFirstTokenTimeoutError, match="time to first token timeout"):
        list(
            llm_client.generate_text_stream(
                prompt="test",
                model=model,
                first_token_timeout=0.2,  # Too short for first token (0.3s)
                inactivity_timeout=0.3,
            )
        )

    # First token should not have been generated since timeout occurred before yield
    assert len(tokens_generated) == 0

    # Should timeout on inactivity if inactivity timeout is too low
    tokens_generated = []
    with pytest.raises(LlmStreamInactivityTimeoutError, match="inactivity timeout"):
        list(
            llm_client.generate_text_stream(
                prompt="test",
                model=model,
                first_token_timeout=0.5,  # Enough for first token
                inactivity_timeout=0.1,  # Too short for second token (0.2s)
            )
        )

    # Only first token should have been generated before timeout
    assert len(tokens_generated) == 1
    assert tokens_generated[0][0] == "first"


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
