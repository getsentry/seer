import json

import pytest
from pydantic import BaseModel

from seer.automation.agent.client import (
    AnthropicProvider,
    GeminiProvider,
    LlmClient,
    LlmGenerateStructuredResponse,
    LlmGenerateTextResponse,
    LlmProviderType,
    Message,
    OpenAiProvider,
    ToolCall,
    Usage,
)
from seer.automation.agent.models import LlmRefusalError
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

    assert returned_system_prompt[0].text == system_prompt


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
    content_chunks = [item for item in stream_items if isinstance(item, str)]
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
    content_chunks = [item for item in stream_items if isinstance(item, str)]
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
    content_chunks = [item for item in stream_items if isinstance(item, str)]
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
    model = GeminiProvider.model("gemini-2.0-flash-exp")

    response = llm_client.generate_text(
        prompt="Say hello",
        model=model,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content.strip() == "Hello! How can I help you today?"
    assert response.message.role == "assistant"
    assert response.metadata.model == "gemini-2.0-flash-exp"
    assert response.metadata.provider_name == LlmProviderType.GEMINI
    assert response.metadata.usage == Usage(completion_tokens=10, prompt_tokens=2, total_tokens=12)


@pytest.mark.vcr()
def test_gemini_generate_text_with_tools():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-exp")

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
        prompt="Please invoke test_function with x = 'i love poetry' and write a haiku about the night sky.",
        model=model,
        tools=tools,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert len(response.message.content) > 0
    assert response.message.role == "tool_use"
    assert response.message.tool_calls == [
        ToolCall(
            function="test_function",
            args='{"x": "i love poetry"}',
        ),
    ]


@pytest.mark.vcr()
def test_gemini_generate_structured():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-exp")

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
    assert response.metadata.model == "gemini-2.0-flash-exp"
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
    assert message_dicts[0].parts[0].text == "Hello"
    assert message_dicts[1].role == "model"
    assert message_dicts[1].parts[0].text == "Hi there!"
    assert message_dicts[2].role == "user"
    assert message_dicts[2].parts[0].text == prompt

    assert tool_dicts
    if tool_dicts:
        assert len(tool_dicts) == 1
        assert tool_dicts[0].function_declarations[0].name == "test_function"
        assert tool_dicts[0].function_declarations[0].description == "A test function"
        assert tool_dicts[0].function_declarations[0].parameters.properties["x"].type == "STRING"


@pytest.mark.vcr()
def test_gemini_generate_text_stream():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-exp")

    stream_items = list(
        llm_client.generate_text_stream(
            prompt="Say hello",
            model=model,
        )
    )

    # Check that we got content chunks and usage
    content_chunks = [item for item in stream_items if isinstance(item, str)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

    assert len(content_chunks) > 0
    assert "".join(content_chunks).strip() == "Hello! How can I help you today?"
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
    model = GeminiProvider.model("gemini-2.0-flash-exp")

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
            prompt="Please invoke test_function with x = 'i love poetry' and write a haiku about the night sky.",
            model=model,
            tools=tools,
        )
    )

    # Check that we got content chunks, tool calls and usage
    content_chunks = [item for item in stream_items if isinstance(item, str)]
    tool_calls = [item for item in stream_items if isinstance(item, ToolCall)]
    usage_items = [item for item in stream_items if isinstance(item, Usage)]

    assert len(content_chunks) > 0
    assert len(tool_calls) == 1
    assert tool_calls[0].function == "test_function"
    assert tool_calls[0].args == '{"x": "i love poetry"}'
    assert len(usage_items) == 1


def test_construct_message_from_stream_gemini():
    llm_client = LlmClient()
    model = GeminiProvider.model("gemini-2.0-flash-exp")

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


@pytest.mark.vcr()
def test_gemini_generate_text_from_web_search():
    llm_client = LlmClient()
    model = GeminiProvider(model_name="gemini-2.0-flash-exp")

    response = llm_client.generate_text_from_web_search(
        prompt="What year is it?",
        model=model,
    )

    assert isinstance(response, str)
    assert "2024" in response
