from unittest.mock import MagicMock, patch

import pytest
from johen import generate

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, OpenAiProvider
from seer.automation.agent.models import (
    LlmGenerateTextResponse,
    LlmResponseMetadata,
    Message,
    ToolCall,
    Usage,
)
from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    DefaultStep,
)
from seer.automation.state import LocalMemoryState


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


def test_should_continue_returns_false_when_max_iterations_reached(autofix_agent, run_config):
    autofix_agent.iterations = run_config.max_iterations
    autofix_agent.memory = [Message(role="user", content="You're great when you work")]
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    assert not autofix_agent.should_continue(run_config)


def test_should_continue_returns_false_with_stop_message(autofix_agent, run_config):
    run_config.stop_message = "STOP"
    autofix_agent.memory = [Message(role="assistant", content="Let's STOP here")]
    autofix_agent.iterations = 1
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    assert not autofix_agent.should_continue(run_config)


def test_should_continue_returns_false_when_assistant_message_has_no_tool_calls(
    autofix_agent, run_config
):
    autofix_agent.memory = [Message(role="assistant", content="All done!")]
    autofix_agent.iterations = 1
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    assert not autofix_agent.should_continue(run_config)


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
@patch.object(AutofixContext, "get_repo_client", return_value=MagicMock())
@patch.object(AutofixContext, "get_file_contents", return_value="print('mock file content')")
def test_run_iteration_with_insight_sharing(
    mock_get_file_contents, mock_get_repo_client, autofix_agent, run_config
):
    autofix_agent.config.interactive = True
    autofix_agent.memory = [
        Message(
            role="user",
            content="My code has a capitalization error, first write an explanation of the error and then use the tools provided to fix it: ```python\nprint('hello World!')\n```",
        )
    ]
    autofix_agent.tools = [
        FunctionTool(
            name="fix_capitalization",
            description="Fix the capitalization of the code",
            parameters=[],
            fn=lambda: None,
        )
    ]
    with autofix_agent.context.state.update() as state:
        state.request.options.disable_interactivity = False
        state.steps = [
            DefaultStep(
                status=AutofixStatus.NEED_MORE_INFORMATION,
                key="root_cause_analysis_processing",
                title="Test",
                id="id",
            )
        ]

    with autofix_agent.manage_run():
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
                key="root_cause_analysis_processing",
                title="Fixing a bug",
                insights=[next(generate(InsightSharingOutput))],
                id="id",
            )
        ]

    initial_insights_count = len(autofix_agent.context.state.get().steps[-1].insights)
    autofix_agent.share_insights(
        "Thinking about the solution", autofix_agent.context.state, 0, "id"
    )
    final_insights_count = len(autofix_agent.context.state.get().steps[-1].insights)

    assert initial_insights_count == final_insights_count


@patch("seer.automation.autofix.autofix_agent.AutofixAgent.get_completion")
@pytest.mark.vcr()
def test_run_iteration_no_help_prompt_when_not_needed(
    mock_get_completion, interactive_autofix_agent, run_config
):
    mock_completion = MagicMock(
        message=Message(role="assistant", content="Thinking about the solution...")
    )
    mock_get_completion.return_value = mock_completion

    with interactive_autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
        state.request.options.disable_interactivity = False

    with interactive_autofix_agent.manage_run():
        interactive_autofix_agent.iterations = 4  # not divisible by 6 and not near max
        interactive_autofix_agent.run_iteration(run_config)

    assert len(interactive_autofix_agent.memory) == 1
    assert interactive_autofix_agent.memory[0] == mock_completion.message


@pytest.mark.vcr()
def test_get_completion_interrupts_with_queued_messages(interactive_autofix_agent, run_config):
    # Set up queued user messages
    with interactive_autofix_agent.context.state.update() as state:
        state.steps.append(
            DefaultStep(
                status=AutofixStatus.PROCESSING,
                key="test",
                title="Test",
                queued_user_messages=["User interruption"],
            )
        )

    # Call get_completion and verify it returns None due to interruption
    result = interactive_autofix_agent.get_completion(run_config)
    assert result is None


@patch("seer.automation.autofix.autofix_agent.AutofixAgent._get_completion")
def test_get_completion_retries_for_retryable_errors(
    mock_get_completion, autofix_agent, run_config
):
    """Test that get_completion retries when AnthropicProvider raises retryable errors."""
    from anthropic import AnthropicError

    from seer.automation.agent.client import AnthropicProvider
    from seer.automation.agent.models import LlmStreamFirstTokenTimeoutError, Message, Usage

    # Set up the run config to use the AnthropicProvider
    run_config.model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    # Set up the mock to raise an exception for the first two calls and succeed on the third call
    mock_response = LlmGenerateTextResponse(
        message=Message(role="assistant", content="Response after retry"),
        metadata=LlmResponseMetadata(
            model=run_config.model.model_name,
            provider_name=run_config.model.provider_name,
            usage=Usage(completion_tokens=10, prompt_tokens=5, total_tokens=15),
        ),
    )

    # First call raises AnthropicError with overloaded_error message (which is retryable)
    # Second call raises LlmStreamFirstTokenTimeoutError (which is retryable)
    # Third call succeeds
    mock_get_completion.side_effect = [
        AnthropicError("Your request failed due to overloaded_error"),
        LlmStreamFirstTokenTimeoutError("Stream time to first token timeout after 40.0 seconds"),
        mock_response,
    ]

    # Call the get_completion method
    result = autofix_agent.get_completion(run_config, max_tries=4)

    # Verify that get_completion was called multiple times due to retries
    assert mock_get_completion.call_count == 3

    # Verify the final result is correct
    assert result == mock_response
    assert result.message.content == "Response after retry"
    assert result.metadata.usage.total_tokens == 15


@patch("seer.automation.autofix.autofix_agent.AutofixAgent._get_completion")
def test_get_completion_does_not_retry_for_non_retryable_errors(
    mock_get_completion, autofix_agent, run_config
):
    """Test that get_completion doesn't retry for non-retryable errors."""
    from anthropic import AnthropicError

    from seer.automation.agent.client import AnthropicProvider

    # Set up the run config to use the AnthropicProvider
    run_config.model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

    # Set up the mock to raise a non-retryable exception
    mock_get_completion.side_effect = AnthropicError("Some other non-retryable error")

    # Call the get_completion method and expect the exception to be propagated
    with pytest.raises(AnthropicError, match="Some other non-retryable error"):
        autofix_agent.get_completion(run_config, max_tries=4)

    # Verify get_completion was called only once (no retries)
    assert mock_get_completion.call_count == 1


@patch("seer.automation.autofix.autofix_agent.AutofixAgent._get_completion")
def test_get_completion_retries_truncates_anthropic_input_too_long(
    mock_get_completion,
    autofix_agent,
    run_config,
):
    """Test get_completion retries, truncates messages, and calls again for Anthropic input too long error using real helper methods."""
    run_config.model = AnthropicProvider.model("claude-3-7-sonnet@20250219")
    input_too_long_exception = Exception(
        "Error message: Prompt is too long and exceeds the context window limit."
    )
    mock_success_response = LlmGenerateTextResponse(
        message=Message(role="assistant", content="Success after truncation"),
        metadata=LlmResponseMetadata(
            model=run_config.model.model_name,
            provider_name=run_config.model.provider_name,
            usage=Usage(completion_tokens=10, prompt_tokens=5, total_tokens=15),
        ),
    )

    mock_get_completion.side_effect = [input_too_long_exception, mock_success_response]

    initial_memory = [
        Message(role="user", content="initial prompt"),
        Message(
            role="assistant",
            content="thinking...",
            tool_calls=[ToolCall(id="t1", function="foo", args="{}")],
        ),
        Message(role="tool", content="Result of tool 1 - short", tool_call_id="t1"),
        Message(
            role="tool",
            content="Result of tool 2 - this one is very long and should be considered for truncation",
            tool_call_id="t2",
        ),
        Message(role="tool", content="Result of tool 3 - medium length content", tool_call_id="t3"),
        Message(role="tool", content="Result of tool 4 - another short one", tool_call_id="t4"),
    ]
    autofix_agent.memory = initial_memory

    # Calculate the expected memory after truncation (truncates largest 3 tool messages)
    expected_truncated_memory = [
        Message(role="user", content="initial prompt"),
        Message(
            role="assistant",
            content="thinking...",
            tool_calls=[ToolCall(id="t1", function="foo", args="{}")],
        ),
        Message(role="tool", content="Result of tool 1 - short", tool_call_id="t1"),
        Message(role="tool", content="[omitted for brevity]", tool_call_id="t2"),
        Message(role="tool", content="[omitted for brevity]", tool_call_id="t3"),
        Message(role="tool", content="[omitted for brevity]", tool_call_id="t4"),
    ]

    result = autofix_agent.get_completion(run_config)

    assert mock_get_completion.call_count == 2, "_get_completion should be called twice"

    first_call_args = mock_get_completion.call_args_list[0]
    second_call_args = mock_get_completion.call_args_list[1]

    assert first_call_args.args[0] == run_config
    assert first_call_args.kwargs["messages"] == initial_memory, "First call used original memory"

    assert second_call_args.args[0] == run_config
    assert (
        second_call_args.kwargs["messages"] == expected_truncated_memory
    ), "Second call did not use the correctly truncated memory"

    assert result == mock_success_response, "Final result should be the success response"


def test_run_iteration_skips_duplicate_tool_calls(autofix_agent, run_config):
    """
    If a tool call with the same function and args was already called, AutofixAgent should not call it again,
    and should append a message indicating the tool was not called again.
    """
    # Prepare memory with a previous tool call result
    prev_tool_call = ToolCall(id="t1", function="fix_bug", args='{"line": 42}')
    autofix_agent.memory = [
        Message(role="user", content="Please fix the bug on line 42."),
        Message(role="assistant", content="Calling fix_bug tool", tool_calls=[prev_tool_call]),
        Message(role="tool", content="Bug fixed!", tool_call_id="t1", tool_call_function="fix_bug"),
    ]
    # Prepare a completion that tries to call the same tool with the same args
    duplicate_tool_call = ToolCall(id="t2", function="fix_bug", args='{"line": 42}')
    completion_message = Message(
        role="assistant",
        content="Calling fix_bug tool again",
        tool_calls=[duplicate_tool_call],
    )
    completion = LlmGenerateTextResponse(
        message=completion_message,
        metadata=LlmResponseMetadata(
            model=run_config.model.model_name,
            provider_name=run_config.model.provider_name,
            usage=Usage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    )
    autofix_agent.get_completion = lambda rc: completion
    autofix_agent.call_tool = MagicMock()

    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    with autofix_agent.manage_run():
        autofix_agent.run_iteration(run_config)
    # The duplicate tool call should not have triggered call_tool
    autofix_agent.call_tool.assert_not_called()
    # The last message should indicate the tool was not called again
    assert autofix_agent.memory[-1].role == "tool"
    assert (
        "not called again" in autofix_agent.memory[-1].content
        and "already called" in autofix_agent.memory[-1].content
    )
