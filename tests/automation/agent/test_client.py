from unittest.mock import Mock, patch

import anthropic
import pytest

from seer.automation.agent.client import ClaudeClient, GptClient, LlmClient
from seer.automation.agent.models import Message, ToolCall


@pytest.fixture
def mock_openai_client():
    with patch("openai.Client") as mock:
        yield mock


@pytest.fixture
def mock_anthropic_client():
    with patch("anthropic.AnthropicVertex") as mock:
        yield mock


def test_gpt_client_completion(mock_openai_client):
    client = GptClient()
    mock_response = Mock(
        choices=[Mock(message=Mock(content="Test response", role="assistant", tool_calls=None))],
        usage=Mock(completion_tokens=10, prompt_tokens=20, total_tokens=30),
    )
    mock_openai_client.chat.completions.create.return_value = mock_response
    client.openai_client = mock_openai_client

    messages = [Message(role="user", content="Test message")]
    message, usage = client.completion(messages)

    assert mock_openai_client.chat.completions.create.call_count == 1
    assert message.content == "Test response"
    assert message.role == "assistant"
    assert usage.completion_tokens == 10
    assert usage.prompt_tokens == 20
    assert usage.total_tokens == 30


def test_gpt_client_json_completion(mock_openai_client):
    client = GptClient()
    mock_response = Mock(
        choices=[Mock(message=Mock(content='{"key": "value"}', role="assistant", tool_calls=None))],
        usage=Mock(completion_tokens=10, prompt_tokens=20, total_tokens=30),
    )
    mock_openai_client.chat.completions.create.return_value = mock_response
    client.openai_client = mock_openai_client

    messages = [Message(role="user", content="Get me some JSON")]
    result, message, usage = client.json_completion(messages, model="gpt-4")

    assert mock_openai_client.chat.completions.create.call_count == 1
    assert result == {"key": "value"}
    assert message.content == '{"key": "value"}'
    assert usage.total_tokens == 30


def test_claude_client_completion(mock_anthropic_client):
    client = ClaudeClient()
    mock_response = anthropic.types.Message(
        id="id",
        type="message",
        content=[anthropic.types.TextBlock(type="text", text="Claude response")],
        role="assistant",
        model="some-claude-model",
        usage=anthropic.types.Usage(input_tokens=20, output_tokens=10),
    )
    mock_anthropic_client.messages.create.return_value = mock_response
    client.anthropic_client = mock_anthropic_client

    messages = [Message(role="user", content="Hello Claude")]
    message, usage = client.completion(messages)

    assert mock_anthropic_client.messages.create.call_count == 1
    assert message.content == "Claude response"
    assert message.role == "assistant"
    assert usage.completion_tokens == 20
    assert usage.prompt_tokens == 10
    assert usage.total_tokens == 30


def test_claude_client_tool_use(mock_anthropic_client):
    client = ClaudeClient()
    mock_response = anthropic.types.Message(
        id="id",
        type="message",
        content=[
            anthropic.types.ToolUseBlock(
                id="tool1",
                name="search",
                input={"query": "test"},
                type="tool_use",
            )
        ],
        model="some-claude-model",
        role="assistant",
        usage=anthropic.types.Usage(input_tokens=20, output_tokens=10),
    )
    mock_anthropic_client.messages.create.return_value = mock_response
    client.anthropic_client = mock_anthropic_client

    messages = [Message(role="user", content="Use a tool")]
    message, usage = client.completion(messages)

    assert mock_anthropic_client.messages.create.call_count == 1
    assert message.role == "tool_use"
    assert message.tool_calls[0].id == "tool1"
    assert message.tool_calls[0].function == "search"
    assert message.tool_calls[0].args == '{"query": "test"}'


def test_claude_client_json_completion(mock_anthropic_client):
    client = ClaudeClient()
    mock_response = anthropic.types.Message(
        id="id",
        type="message",
        content=[anthropic.types.TextBlock(type="text", text='{"key": "value"}')],
        role="assistant",
        model="some-claude-model",
        usage=anthropic.types.Usage(input_tokens=20, output_tokens=10),
    )
    mock_anthropic_client.messages.create.return_value = mock_response
    client.anthropic_client = mock_anthropic_client

    messages = [Message(role="user", content="Get me some JSON")]
    result, message, usage = client.json_completion(messages, model="claude-3")

    assert mock_anthropic_client.messages.create.call_count == 1
    assert result == {"key": "value"}
    assert message.content == '{"key": "value"}'
    assert usage.total_tokens == 30


def test_format_messages_for_claude_input():
    client = ClaudeClient()

    input_messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="tool", content="Tool result", tool_call_id="tool1"),
        Message(
            role="tool_use",
            tool_calls=[ToolCall(id="tool2", function="search", args='{"query": "test"}')],
        ),
    ]

    formatted_messages = client._format_messages_for_claude_input(input_messages)

    assert len(formatted_messages) == 4
    assert formatted_messages[0] == {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    assert formatted_messages[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}],
    }
    assert formatted_messages[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "content": "Tool result", "tool_use_id": "tool1"}],
    }
    assert formatted_messages[3] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "tool2", "name": "search", "input": {"query": "test"}}
        ],
    }


def test_format_claude_response_to_message():
    client = ClaudeClient()

    # Test text response
    text_response = anthropic.types.Message(
        id="id",
        type="message",
        content=[anthropic.types.TextBlock(type="text", text="Hello, how can I help you?")],
        role="assistant",
        model="some-claude-model",
        usage=anthropic.types.Usage(input_tokens=20, output_tokens=10),
    )
    text_message = client._format_claude_response_to_message(text_response)
    assert text_message.role == "assistant"
    assert text_message.content == "Hello, how can I help you?"
    assert text_message.tool_calls is None

    # Test tool use response
    tool_use_response = anthropic.types.Message(
        role="assistant",
        type="message",
        id="id",
        model="some-claude-model",
        content=[
            anthropic.types.ToolUseBlock(
                type="tool_use", id="tool1", name="search", input={"query": "Python programming"}
            )
        ],
        usage=anthropic.types.Usage(input_tokens=20, output_tokens=10),
    )
    tool_use_message = client._format_claude_response_to_message(tool_use_response)
    assert tool_use_message.role == "tool_use"
    assert tool_use_message.content is None
    assert len(tool_use_message.tool_calls) == 1
    assert tool_use_message.tool_calls[0].id == "tool1"
    assert tool_use_message.tool_calls[0].function == "search"
    assert tool_use_message.tool_calls[0].args == '{"query": "Python programming"}'
    assert tool_use_message.tool_call_id == "tool1"

    # Test mixed response
    mixed_response = anthropic.types.Message(
        role="assistant",
        type="message",
        id="id",
        model="some-claude-model",
        content=[
            anthropic.types.TextBlock(type="text", text="Here's what I found:"),
            anthropic.types.ToolUseBlock(
                type="tool_use", id="tool2", name="search", input={"query": "AI developments"}
            ),
        ],
        usage=anthropic.types.Usage(input_tokens=20, output_tokens=10),
    )
    mixed_message = client._format_claude_response_to_message(mixed_response)
    assert mixed_message.role == "tool_use"
    assert mixed_message.content == "Here's what I found:"
    assert len(mixed_message.tool_calls) == 1
    assert mixed_message.tool_calls[0].id == "tool2"
    assert mixed_message.tool_calls[0].function == "search"
    assert mixed_message.tool_calls[0].args == '{"query": "AI developments"}'
    assert mixed_message.tool_call_id == "tool2"


def test_llm_client_abstract_methods():
    with pytest.raises(TypeError):
        LlmClient()
