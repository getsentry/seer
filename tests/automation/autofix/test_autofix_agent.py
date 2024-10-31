import pytest
from johen import generate

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import OpenAiProvider
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.component import FormatStepOnePrompt
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    DefaultStep,
)
from seer.automation.state import LocalMemoryState
from seer.dependency_injection import Module


@pytest.fixture
def context():
    request = next(generate(AutofixRequest))
    continuation = AutofixContinuation(request=request)
    state = LocalMemoryState(val=continuation)
    return AutofixContext(state=state, event_manager=AutofixEventManager(state=state))


@pytest.fixture
def autofix_agent(context: AutofixContext):
    config = AgentConfig()
    return AutofixAgent(config=config, context=context, name="TestAutofixAgent")


@pytest.fixture
def interactive_autofix_agent(context: AutofixContext):
    config = AgentConfig(interactive=True)
    return AutofixAgent(config=config, context=context, name="TestAutofixAgent")


@pytest.fixture
def run_config():
    return RunConfig(
        system_prompt="You are a helpful assistant for fixing code.",
        prompt="Fix this bug.",
        model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
        temperature=0.0,
        run_name="Test Autofix Run",
    )


def test_should_continue_waiting_for_user_response(
    autofix_agent: AutofixAgent, run_config: RunConfig
):
    with autofix_agent.context.state.update() as state:
        state.steps = [
            DefaultStep(status=AutofixStatus.WAITING_FOR_USER_RESPONSE, key="test", title="Test")
        ]
    assert not autofix_agent.should_continue(run_config)


def test_should_continue_normal_case(autofix_agent: AutofixAgent, run_config: RunConfig):
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    assert autofix_agent.should_continue(run_config)


@pytest.mark.vcr()
def test_run_iteration_with_queued_user_messages(
    interactive_autofix_agent,
    run_config,
):
    with interactive_autofix_agent.context.state.update() as state:
        state.steps.append(
            DefaultStep(
                status=AutofixStatus.PROCESSING,
                key="test",
                title="Test",
                queued_user_messages=["User input"],
            )
        )

    with interactive_autofix_agent.manage_run():
        interactive_autofix_agent.run_iteration(run_config)

    assert len(interactive_autofix_agent.memory) == 2
    assert interactive_autofix_agent.memory[0] == Message(role="user", content="User input")
    assert interactive_autofix_agent.memory[1].content.startswith(
        "It seems like you might be looking for help"
    )


@pytest.mark.vcr()
def test_run_iteration_with_insight_sharing(autofix_agent, run_config):
    autofix_agent.config.interactive = True
    with autofix_agent.context.state.update() as state:
        state.request.options.disable_interactivity = False
        state.steps = [
            DefaultStep(status=AutofixStatus.NEED_MORE_INFORMATION, key="test", title="Test")
        ]

    with (
        Module().constant(
            FormatStepOnePrompt,
            FormatStepOnePrompt("Say something insightful like, 'This needs more abstraction'"),
        ),
        autofix_agent.manage_run(),
    ):
        autofix_agent.run_iteration(run_config)

    assert autofix_agent.context.state.get().usage.total_tokens > 0
    assert len(autofix_agent.context.state.get().steps[-1].insights) > 0


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


@pytest.mark.vcr()
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
    autofix_agent.share_insights("Thinking about the solution", 0, autofix_agent.context.state, 0)
    final_insights_count = len(autofix_agent.context.state.get().steps[-1].insights)

    assert initial_insights_count == final_insights_count
