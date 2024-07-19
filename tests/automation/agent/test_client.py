from unittest.mock import Mock, patch

import anthropic
import pytest

from seer.automation.agent.client import ClaudeClient, GptClient, LlmClient
from seer.automation.agent.models import Message


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
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    messages = [Message(role="user", content="Test message")]
    message, usage = client.completion(messages)

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
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    messages = [Message(role="user", content="Get me some JSON")]
    result, message, usage = client.json_completion(messages, model="gpt-4")

    assert result == {"key": "value"}
    assert message.content == '{"key": "value"}'
    assert usage.total_tokens == 30


def test_gpt_client_error_handling(mock_openai_client):
    client = GptClient()
    mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API Error")

    messages = [Message(role="user", content="Test message")]
    with pytest.raises(Exception, match="API Error"):
        client.completion(messages)


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
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    messages = [Message(role="user", content="Hello Claude")]
    message, usage = client.completion(messages)

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
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    messages = [Message(role="user", content="Use a tool")]
    message, usage = client.completion(messages)

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
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    messages = [Message(role="user", content="Get me some JSON")]
    result, message, usage = client.json_completion(messages, model="claude-3")

    assert result == {"key": "value"}
    assert message.content == '{"key": "value"}'
    assert usage.total_tokens == 30


def test_claude_client_error_handling(mock_anthropic_client):
    client = ClaudeClient()
    mock_anthropic_client.return_value.messages.create.side_effect = Exception("API Error")

    messages = [Message(role="user", content="Test message")]
    with pytest.raises(Exception, match="API Error"):
        client.completion(messages)


def test_llm_client_abstract_methods():
    with pytest.raises(TypeError):
        LlmClient()
