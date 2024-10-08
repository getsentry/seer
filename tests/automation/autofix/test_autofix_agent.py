from unittest.mock import MagicMock, Mock, patch

import pytest
from johen import generate

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import OpenAiProvider
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    DefaultStep,
)
from seer.automation.state import LocalMemoryState


@pytest.fixture
def mock_context():
    request = next(generate(AutofixRequest))
    continuation = AutofixContinuation(request=request)
    state = LocalMemoryState(val=continuation)
    return MagicMock(state=state)


@pytest.fixture
def mock_llm_client():
    return Mock()


@pytest.fixture
def autofix_agent(mock_context, mock_llm_client):
    config = AgentConfig()
    return AutofixAgent(config=config, context=mock_context, name="TestAutofixAgent")


@pytest.fixture
def interactive_autofix_agent(mock_context, mock_llm_client):
    config = AgentConfig(interactive=True)
    return AutofixAgent(config=config, context=mock_context, name="TestAutofixAgent")


@pytest.fixture
def run_config():
    return RunConfig(
        system_prompt="You are a helpful assistant for fixing code.",
        prompt="Fix this bug.",
        model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
        temperature=0.0,
        run_name="Test Autofix Run",
    )


def test_autofix_agent_initialization(autofix_agent, mock_context):
    assert autofix_agent.name == "TestAutofixAgent"
    assert autofix_agent.context == mock_context
    assert autofix_agent.iterations == 0
    assert autofix_agent.memory == []


def test_should_continue_waiting_for_user_response(autofix_agent, run_config):
    with autofix_agent.context.state.update() as state:
        state.steps = [
            DefaultStep(status=AutofixStatus.WAITING_FOR_USER_RESPONSE, key="test", title="Test")
        ]
    assert not autofix_agent.should_continue(run_config)


def test_should_continue_normal_case(autofix_agent, run_config):
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    assert autofix_agent.should_continue(run_config)


@patch("seer.automation.autofix.autofix_agent.AutofixAgent.get_completion")
def test_run_iteration_with_queued_user_messages(
    mock_get_completion,
    interactive_autofix_agent,
    run_config,
):
    mock_completion = MagicMock(
        message=Message(role="assistant", content="Thinking about the solution...")
    )
    mock_get_completion.return_value = mock_completion

    with interactive_autofix_agent.context.state.update() as state:
        state.steps.append(
            DefaultStep(
                status=AutofixStatus.PROCESSING,
                key="test",
                title="Test",
                queued_user_messages=["User input"],
            )
        )

    interactive_autofix_agent.run_iteration(run_config)

    assert len(interactive_autofix_agent.memory) == 2
    assert interactive_autofix_agent.memory[0] == Message(role="user", content="User input")
    assert interactive_autofix_agent.memory[1] == mock_completion.message


@patch("seer.automation.autofix.autofix_agent.AutofixAgent.get_completion")
@patch("seer.automation.autofix.autofix_agent.AutofixAgent.run_in_thread")
def test_run_iteration_with_insight_sharing(
    mock_run_in_thread, mock_get_completion, autofix_agent, run_config
):
    mock_completion = MagicMock()
    mock_completion.message.content = "Thinking about the solution..."
    mock_get_completion.return_value = mock_completion
    autofix_agent.config.interactive = True
    with autofix_agent.context.state.update() as state:
        state.request.options.disable_interactivity = False

    autofix_agent.run_iteration(run_config)

    mock_run_in_thread.assert_called_once()
    assert mock_run_in_thread.call_args[1]["func"] == autofix_agent.share_insights


@patch("seer.automation.autofix.autofix_agent.AutofixAgent.get_completion")
@patch("seer.automation.autofix.autofix_agent.AutofixAgent.call_tool")
def test_run_iteration_with_tool_calls(
    mock_call_tool, mock_get_completion, autofix_agent, run_config
):
    mock_completion = MagicMock()
    mock_completion.message.tool_calls = [MagicMock(), MagicMock()]
    mock_get_completion.return_value = mock_completion
    mock_call_tool.return_value = MagicMock()

    autofix_agent.run_iteration(run_config)

    assert mock_call_tool.call_count == 2
    assert len(autofix_agent.memory) == 3  # Completion message + 2 tool responses


def test_use_user_messages(autofix_agent):
    with autofix_agent.context.state.update() as state:
        state.steps = [
            DefaultStep(
                status=AutofixStatus.PROCESSING,
                key="test",
                title="Test",
                queued_user_messages=["User input 1"],
            )
        ]
    autofix_agent.memory = [Message(role="assistant", content="Previous response")]

    autofix_agent.use_user_messages()

    assert len(autofix_agent.memory) == 2
    assert autofix_agent.memory[-2].role == "assistant"
    assert autofix_agent.memory[-2].content == "Previous response"
    assert autofix_agent.memory[-1].role == "user"
    assert autofix_agent.memory[-1].content == "User input 1"


@patch("seer.automation.autofix.autofix_agent.InsightSharingComponent")
def test_share_insights(mock_insight_sharing_component, autofix_agent):
    mock_component = MagicMock()
    mock_insight_sharing_component.return_value = mock_component
    mock_component.invoke.return_value = MagicMock()

    autofix_agent.memory = [Message(role="user", content="Fix this bug")]

    with autofix_agent.context.state.update() as state:
        state.steps = [
            DefaultStep(
                status=AutofixStatus.PROCESSING, key="test", title="Fixing a bug", insights=[]
            )
        ]

    autofix_agent.share_insights(autofix_agent.context, "Thinking about the solution", 0)

    mock_component.invoke.assert_called_once()
    assert len(autofix_agent.context.state.get().steps[-1].insights) == 1


def test_share_insights_no_new_insights(autofix_agent):
    with autofix_agent.context.state.update() as state:
        state.steps = [
            DefaultStep(
                status=AutofixStatus.PROCESSING,
                key="test",
                title="Fixing a bug",
                insights=[next(generate(InsightSharingOutput))],
            )
        ]

    initial_insights_count = len(autofix_agent.context.state.get().steps[-1].insights)
    autofix_agent.share_insights(autofix_agent.context, "Thinking about the solution", 0)
    final_insights_count = len(autofix_agent.context.state.get().steps[-1].insights)

    assert initial_insights_count == final_insights_count
