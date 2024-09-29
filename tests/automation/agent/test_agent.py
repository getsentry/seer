from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from seer.automation.agent.agent import (
    AgentConfig,
    ClaudeAgent,
    GptAgent,
    LlmAgent,
    MaxIterationsReachedException,
)
from seer.automation.agent.client import ClaudeClient, GptClient, LlmClient, T
from seer.automation.agent.models import Message, ToolCall, Usage
from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import InsightSharingRequest
from seer.automation.autofix.models import DefaultStep
from seer.dependency_injection import resolve


class TestLlmAgent:
    @pytest.fixture
    def config(self):
        return AgentConfig()

    @pytest.fixture
    def client(self):
        class TestClient(LlmClient):
            def completion(
                self,
                messages: list[Message],
                model: str,
                system_prompt: str | None = None,
                tools: list[FunctionTool] | None = ...,
                response_format: dict | None = None,
            ) -> tuple[Message, Usage]:
                pass

            def completion_with_parser(
                self,
                messages: list[Message],
                parser: Callable[[str | None], T],
                model: str,
                system_prompt: str | None = None,
                tools: list[FunctionTool] | None = None,
                response_format: dict | None = None,
            ) -> tuple[T, Message, Usage]:
                pass

            def json_completion(
                self, messages: list[Message], model: str, system_prompt: str | None = None
            ) -> tuple[dict[str, Any] | None, Message, Usage]:
                pass

        return TestClient()

    @pytest.fixture
    def agent(self, config, client):
        class TestAgent(LlmAgent):
            def run_iteration(self):
                pass

        return TestAgent(config=config, client=client)

    def test_should_continue(self, agent: LlmAgent):
        assert agent.should_continue()  # Initial state

        agent.iterations = agent.config.max_iterations
        agent.memory = [Message(role="assistant", content="STOP")]
        assert not agent.should_continue()  # Max iterations reached

        agent.iterations = 1
        agent.config.stop_message = "STOP"
        assert not agent.should_continue()  # Stop message found

    def test_add_user_message(self, agent: LlmAgent):
        agent.add_user_message("Test message")
        assert len(agent.memory) == 1
        assert agent.memory[0].role == "user"
        assert agent.memory[0].content == "Test message"

    def test_get_last_message_content(self, agent: LlmAgent):
        assert agent.get_last_message_content() is None  # Empty memory
        agent.memory = [Message(role="user", content="Test")]
        assert agent.get_last_message_content() == "Test"

    def test_call_tool(self, agent: LlmAgent):
        mock_tool_fn = MagicMock(return_value="Tool result")
        mock_tool = FunctionTool(
            name="test_tool",
            description="Test tool",
            fn=mock_tool_fn,
            parameters=[],
        )
        mock_tool.name = "test_tool"

        agent.tools = [mock_tool]

        tool_call = ToolCall(id="123", function="test_tool", args='{"arg": "value"}')
        result = agent.call_tool(tool_call)

        assert isinstance(result, Message)
        assert result.role == "tool"
        assert result.content == "Tool result"
        assert result.tool_call_id == "123"

    def test_get_tool_by_name(self, agent: LlmAgent):
        tool = FunctionTool(
            name="test_tool", description="Test tool", fn=lambda: None, parameters=[]
        )
        agent.tools = [tool]
        assert agent.get_tool_by_name("test_tool") == tool

        with pytest.raises(StopIteration):
            agent.get_tool_by_name("non_existent_tool")

    def test_parse_tool_arguments(self, agent: LlmAgent):
        tool = FunctionTool(
            name="test_tool",
            description="Test tool",
            fn=lambda: None,
            parameters=[{"name": "arg1"}, {"name": "arg2"}],
        )
        args = '{"arg1": "value1", "arg2": "value2", "arg3": "value3"}'
        parsed_args = agent.parse_tool_arguments(tool, args)
        assert parsed_args == {"arg1": "value1", "arg2": "value2"}


class TestGptAgent:
    @pytest.fixture(params=[(GptAgent, GptClient), (ClaudeAgent, ClaudeClient)])
    def agent_and_client_classes(self, request):
        return request.param

    @pytest.fixture
    def config(self):
        return AgentConfig(interactive=True)

    @pytest.fixture
    def agent(self, agent_and_client_classes, config):
        agent_class, _ = agent_and_client_classes
        return agent_class(config)

    @pytest.fixture
    def mock_client(self, agent_and_client_classes):
        _, client_class = agent_and_client_classes
        return resolve(client_class)

    @pytest.fixture
    def mock_context(self):
        context = MagicMock(spec=AutofixContext)
        state = MagicMock()
        state.steps = [DefaultStep(title="Test step title")]
        state.get_all_insights.return_value = []
        state.get_step_description.return_value = "Test step"
        state.request.options.disable_interactivity = False
        context.state = MagicMock()
        context.state.get.return_value = state
        return context

    @patch("seer.automation.agent.agent.InsightSharingComponent")
    @patch("seer.automation.agent.agent.threading.Thread")
    def test_run_iteration(self, mock_thread, mock_insight_sharing, agent, mock_context):
        # Mock the message and usage
        mock_message = Message(role="assistant", content="Test response")
        mock_usage = Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        agent.get_completion = MagicMock(return_value=(mock_message, mock_usage))

        # Mock the insight sharing component
        mock_insight_card = MagicMock()
        mock_insight_sharing_instance = MagicMock()
        mock_insight_sharing_instance.invoke.return_value = mock_insight_card
        mock_insight_sharing.return_value = mock_insight_sharing_instance

        # Mock the threading.Thread
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Run the method
        agent.run_iteration(context=mock_context)

        # Assertions
        assert agent.iterations == 1
        assert len(agent.memory) == 1
        assert agent.memory[0] == mock_message
        assert agent.usage == mock_usage

        # Check if a new thread was created for insight sharing
        mock_thread.assert_called_once()
        assert mock_thread.call_args[1]["target"] == agent.share_insights
        assert isinstance(mock_thread.call_args[1]["args"][0], AutofixContext)
        assert isinstance(mock_thread.call_args[1]["args"][1], str)

        # Check if the thread was started
        mock_thread_instance.start.assert_called_once()

        # To test the share insights method, we need to call it directly
        agent.share_insights(mock_context, "Test response")

        # Check if insight sharing was called
        mock_insight_sharing.assert_called_once_with(mock_context)
        mock_insight_sharing_instance.invoke.assert_called_once()
        assert isinstance(
            mock_insight_sharing_instance.invoke.call_args[0][0], InsightSharingRequest
        )
        mock_context.state.update.assert_called_once()

    def test_run_iteration_with_queued_user_messages(self, agent, mock_client, mock_context):
        # Create a mock step with queued_user_messages as a list of strings
        mock_step = MagicMock(spec=DefaultStep)
        mock_step.queued_user_messages = ["User message 1", "User message 2"]

        # Set the mock step as the last step in the context
        mock_context.state.get().steps = [mock_step]

        agent.use_user_messages = MagicMock()

        mock_message = Message(role="assistant", content="Test response")
        mock_usage = Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        mock_client.completion = MagicMock(return_value=(mock_message, mock_usage))

        # Run the method
        agent.run_iteration(context=mock_context)

        # Assertions
        agent.use_user_messages.assert_called_once_with(mock_context)
        assert agent.iterations == 0  # Should not increment
        assert len(agent.memory) == 0  # Should not add to memory

        # Additional assertion to verify the queued_user_messages
        assert mock_context.state.get().steps[-1].queued_user_messages == [
            "User message 1",
            "User message 2",
        ]

    def test_get_completion(self, agent, mock_client):
        mock_message = Message(role="assistant", content="Test response")
        mock_usage = Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        mock_client.completion = MagicMock(return_value=(mock_message, mock_usage))

        message, usage = agent.get_completion()

        assert message == mock_message
        assert usage == mock_usage

    def test_process_message(self, agent):
        message = Message(role="assistant", content="Test message")
        agent.process_message(message)
        assert len(agent.memory) == 1
        assert agent.memory[0] == message
        assert agent.iterations == 1

    def test_process_tool_calls(self, agent):
        tool_calls = [ToolCall(id="1", function="test_tool", args='{"arg": "value"}')]
        with patch.object(agent, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = Message(
                role="tool", content="Tool result", tool_call_id="1"
            )
            agent.process_tool_calls(tool_calls)
        assert len(agent.memory) == 1
        assert agent.memory[0].role == "tool"
        assert agent.memory[0].content == "Tool result"

    def test_update_usage(self, agent):
        initial_usage = Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        agent.usage = initial_usage
        new_usage = Usage(completion_tokens=5, prompt_tokens=10, total_tokens=15)
        agent.update_usage(new_usage)
        expected_usage = Usage(completion_tokens=15, prompt_tokens=30, total_tokens=45)
        assert agent.usage == expected_usage

    def test_run(self, agent, mock_client):
        mock_message = Message(role="assistant", content="Final response")
        mock_usage = Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        mock_client.completion = MagicMock(return_value=(mock_message, mock_usage))

        result = agent.run("Test prompt")

        assert result == "Final response"
        assert len(agent.memory) > 0
        assert agent.memory[0].role == "user"
        assert agent.memory[0].content == "Test prompt"

    def test_run_max_iterations_exception(self, agent, mock_client):
        agent.config.max_iterations = 1
        tool = FunctionTool(
            name="test_tool", description="Test tool", fn=lambda: None, parameters=[]
        )
        agent.tools = [tool]

        mock_message = Message(
            role="assistant",
            content="Response",
            tool_calls=[ToolCall(id="1", function="test_tool", args="{'arg': 'value'}")],
        )
        mock_usage = Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        mock_client.completion = MagicMock(return_value=(mock_message, mock_usage))

        with pytest.raises(MaxIterationsReachedException):
            agent.run("Test prompt")
