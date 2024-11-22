import json
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from seer.automation.agent.client import (
    AnthropicProvider,
    LlmClient,
    LlmGenerateStructuredResponse,
    LlmGenerateTextResponse,
    LlmProviderType,
    Message,
    OpenAiProvider,
    ToolCall,
    Usage,
)
from seer.automation.agent.tools import FunctionTool


class MockOpenAiFunction(BaseModel):
    name: str
    arguments: str


class MockOpenAiToolCall(BaseModel):
    id: str
    function: MockOpenAiFunction


class MockOpenAIResponse:
    def __init__(
        self,
        *,
        content: str,
        parsed: BaseModel | None = None,
        role: str,
        tool_calls: list[MockOpenAiToolCall] | None = None,
        refusal: str | None = None,
    ):
        self.choices = [
            MagicMock(
                message=MagicMock(
                    parsed=parsed,
                    content=content,
                    role=role,
                    tool_calls=tool_calls,
                    refusal=refusal,
                )
            )
        ]
        self.usage = MagicMock(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        )


class MockContentBlock(BaseModel):
    type: Literal["text", "tool_use"]
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None


class MockAnthropicResponse:
    def __init__(self, content, role, tool_calls=None):
        self.content = [MockContentBlock(type="text", text=content)]
        if tool_calls:
            self.content.extend(
                [
                    MockContentBlock(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function,
                        input=json.loads(tc.args),
                    )
                    for tc in tool_calls
                ]
            )
        self.role = role
        self.usage = MagicMock(
            input_tokens=20,
            output_tokens=10,
        )


@pytest.fixture
def mock_openai_client():
    with patch.object(OpenAiProvider, "get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_anthropic_client():
    with patch.object(AnthropicProvider, "get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def test_openai_generate_text(mock_openai_client):
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse(
        content="Hello, world!", role="assistant"
    )

    response = llm_client.generate_text(
        prompt="Say hello",
        model=model,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Hello, world!"
    assert response.message.role == "assistant"
    assert response.metadata.model == "gpt-3.5-turbo"
    assert response.metadata.provider_name == LlmProviderType.OPENAI
    assert response.metadata.usage == Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    mock_openai_client.chat.completions.create.assert_called_once()


def test_anthropic_generate_text(mock_anthropic_client):
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-sonnet-20240229")

    mock_anthropic_client.messages.create.return_value = MockAnthropicResponse(
        content="Hello, world!", role="assistant"
    )

    response = llm_client.generate_text(
        prompt="Say hello",
        model=model,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Hello, world!"
    assert response.message.role == "assistant"
    assert response.metadata.model == "claude-3-sonnet-20240229"
    assert response.metadata.provider_name == LlmProviderType.ANTHROPIC
    assert response.metadata.usage == Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    mock_anthropic_client.messages.create.assert_called_once()


def test_openai_generate_text_with_tools(mock_openai_client):
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    tool_calls = [
        MockOpenAiToolCall(
            id="1",
            function=MockOpenAiFunction(name="test_function", arguments='{"arg1": "value1"}'),
        )
    ]
    mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse(
        content="Using a tool", role="assistant", tool_calls=tool_calls
    )

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
        prompt="Use a tool",
        model=model,
        tools=tools,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Using a tool"
    assert response.message.role == "assistant"
    assert response.message.tool_calls == [
        ToolCall(id="1", function="test_function", args='{"arg1": "value1"}')
    ]

    mock_openai_client.chat.completions.create.assert_called_once()


def test_anthropic_generate_text_with_tools(mock_anthropic_client):
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-sonnet-20240229")

    tool_calls = [ToolCall(id="1", function="test_function", args='{"arg1": "value1"}')]
    mock_anthropic_client.messages.create.return_value = MockAnthropicResponse(
        content="Using a tool", role="assistant", tool_calls=tool_calls
    )

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
        prompt="Use a tool",
        model=model,
        tools=tools,
    )

    assert isinstance(response, LlmGenerateTextResponse)
    assert response.message.content == "Using a tool"
    assert response.message.role == "tool_use"
    assert response.message.tool_calls == tool_calls

    mock_anthropic_client.messages.create.assert_called_once()


def test_openai_generate_structured(mock_openai_client):
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    class TestStructure(BaseModel):
        name: str
        age: int

    mock_openai_client.beta.chat.completions.parse.return_value = MockOpenAIResponse(
        content='{"name": "John", "age": 30}',
        parsed=TestStructure(name="John", age=30),
        role="assistant",
    )

    response = llm_client.generate_structured(
        prompt="Generate a person",
        model=model,
        response_format=TestStructure,
    )

    assert isinstance(response, LlmGenerateStructuredResponse)
    assert response.parsed == TestStructure(name="John", age=30)
    assert response.metadata.model == "gpt-3.5-turbo"
    assert response.metadata.provider_name == LlmProviderType.OPENAI

    mock_openai_client.beta.chat.completions.parse.assert_called_once()


def test_anthropic_generate_structured():
    llm_client = LlmClient()
    model = AnthropicProvider.model("claude-3-sonnet-20240229")

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


def test_openai_generate_structured_refusal(mock_openai_client):
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-3.5-turbo")

    class TestStructure(BaseModel):
        name: str
        age: int

    mock_openai_client.beta.chat.completions.parse.return_value = MockOpenAIResponse(
        content="I'm sorry, but I can't generate that information.",
        parsed=None,
        role="assistant",
        refusal="I'm sorry, but I can't generate that information.",
    )

    with pytest.raises(Exception) as exc_info:
        llm_client.generate_structured(
            prompt="Generate a person",
            model=model,
            response_format=TestStructure,
        )

    assert str(exc_info.value) == "I'm sorry, but I can't generate that information."

    mock_openai_client.beta.chat.completions.parse.assert_called_once()


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

    assert returned_system_prompt == system_prompt
