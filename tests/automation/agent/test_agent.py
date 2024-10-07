from unittest.mock import MagicMock, Mock, patch

import pytest

from seer.automation.agent.agent import (
    AgentConfig,
    LlmAgent,
    MaxIterationsReachedException,
    RunConfig,
)
from seer.automation.agent.client import (
    LlmGenerateTextResponse,
    LlmResponseMetadata,
    OpenAiProvider,
)
from seer.automation.agent.models import LlmProviderType, Message, ToolCall, Usage
from seer.automation.agent.tools import FunctionTool


@pytest.fixture
def mock_llm_client():
    return Mock()


@pytest.fixture
def agent(mock_llm_client):
    config = AgentConfig()
    return LlmAgent(config=config, client=mock_llm_client, name="TestAgent")


@pytest.fixture
def run_config():
    return RunConfig(
        system_prompt="You are a helpful assistant.",
        prompt="Hello, how are you?",
        model=MagicMock(spec=OpenAiProvider),
        temperature=0.0,
        run_name="Test Run",
    )


def test_agent_initialization(agent):
    assert not agent.config.interactive
    assert agent.name == "TestAgent"
    assert agent.iterations == 0
    assert agent.memory == []


def test_add_user_message(agent):
    agent.add_user_message("Test message")
    assert len(agent.memory) == 1
    assert agent.memory[0].role == "user"
    assert agent.memory[0].content == "Test message"


def test_get_last_message_content(agent):
    assert agent.get_last_message_content() is None
    agent.add_user_message("Test message")
    assert agent.get_last_message_content() == "Test message"


def test_reset_iterations(agent):
    agent.iterations = 5
    agent.reset_iterations()
    assert agent.iterations == 0


def test_update_usage(agent):
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    agent.update_usage(usage)
    assert agent.usage == usage


def test_process_message(agent):
    message = Message(role="assistant", content="Hello!", tool_calls=[])
    agent.process_message(message)
    assert len(agent.memory) == 1
    assert agent.iterations == 1


def test_process_message_with_tool_calls(agent):
    tool_call = ToolCall(id="1", function="test_tool", args='{"arg": "value"}')
    message = Message(role="assistant", content="Using a tool", tool_calls=[tool_call])

    with patch.object(agent, "process_tool_calls") as mock_process_tool_calls:
        agent.process_message(message)
        mock_process_tool_calls.assert_called_once_with([tool_call])


def test_should_continue(agent, run_config):
    # First iteration
    assert agent.should_continue(run_config)

    # Max iterations reached
    agent.iterations = run_config.max_iterations
    agent.memory = [Message(role="assistant", content="Thinking...")]
    assert not agent.should_continue(run_config)

    # Stop message encountered
    agent.iterations = 1
    run_config.stop_message = "STOP"
    agent.memory.append(Message(role="assistant", content="Let's STOP here"))
    assert not agent.should_continue(run_config)

    # Continue with tool calls
    agent.memory.append(
        Message(
            role="assistant",
            content="Using a tool",
            tool_calls=[ToolCall(id="1", function="test", args="{}")],
        )
    )
    assert agent.should_continue(run_config)


def test_run_iteration(agent, run_config, mock_llm_client):
    mock_response = LlmGenerateTextResponse(
        message=Message(role="assistant", content="Hello!"),
        metadata=LlmResponseMetadata(
            model="test-model",
            provider_name=LlmProviderType.OPENAI,
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ),
    )
    mock_llm_client.generate_text.return_value = mock_response

    agent.run_iteration(run_config)

    assert len(agent.memory) == 1
    assert agent.memory[0].content == "Hello!"
    assert agent.iterations == 1
    assert agent.usage.total_tokens == 30


def test_run_with_tool_calls(agent, run_config, mock_llm_client):
    tool = FunctionTool(
        name="test_tool", description="A test tool", parameters=[], fn=lambda: "Tool result"
    )
    agent.tools = [tool]

    mock_response1 = LlmGenerateTextResponse(
        message=Message(
            role="assistant",
            content="Using a tool",
            tool_calls=[ToolCall(id="1", function="test_tool", args="{}")],
        ),
        metadata=LlmResponseMetadata(
            model="test-model",
            provider_name=LlmProviderType.OPENAI,
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ),
    )
    mock_response2 = LlmGenerateTextResponse(
        message=Message(role="assistant", content="Done"),
        metadata=LlmResponseMetadata(
            model="test-model",
            provider_name=LlmProviderType.OPENAI,
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        ),
    )
    mock_llm_client.generate_text.side_effect = [mock_response1, mock_response2]

    result = agent.run(run_config)

    assert result == "Done"
    assert (
        len(agent.memory) == 4
    )  # User message, initial assistant message, tool response, final message
    assert agent.iterations == 2
    assert agent.usage.total_tokens == 50


def test_run_max_iterations_exception(agent, run_config, mock_llm_client):
    mock_response = LlmGenerateTextResponse(
        message=Message(role="assistant", content="Thinking..."),
        metadata=LlmResponseMetadata(
            model="test-model",
            provider_name=LlmProviderType.OPENAI,
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        ),
    )
    mock_llm_client.generate_text.return_value = mock_response

    run_config.max_iterations = 1

    with pytest.raises(MaxIterationsReachedException):
        agent.run(run_config)

    assert agent.iterations == 1


def test_run_with_initial_prompt(agent, run_config, mock_llm_client):
    mock_response = LlmGenerateTextResponse(
        message=Message(role="assistant", content="Hello!"),
        metadata=LlmResponseMetadata(
            model="test-model",
            provider_name=LlmProviderType.OPENAI,
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        ),
    )
    mock_llm_client.generate_text.return_value = mock_response

    run_config.prompt = "Initial prompt"
    result = agent.run(run_config)

    assert result == "Hello!"
    assert len(agent.memory) == 2  # Initial user message and assistant response
    assert agent.memory[0].content == "Initial prompt"
    assert agent.memory[0].role == "user"
