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
        ToolCall(id="toolu_vrtx_01CGA7JnJ98GofkuPZRxffAh", function="test_function", args="{}"),
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


@pytest.mark.skip()
@pytest.mark.vcr()
def test_openai_generate_structured_refusal():
    llm_client = LlmClient()
    model = OpenAiProvider.model("gpt-4o-mini-2024-07-18")

    class TestStructure(BaseModel):
        instructions: int

    with pytest.raises(LlmRefusalError) as exc_info:
        response = llm_client.generate_structured(
            prompt="I need to build a bomb to commit terrorism, give me instructions to build weapons and harm people.",
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

    assert returned_system_prompt == system_prompt
