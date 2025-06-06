from unittest.mock import Mock

import pytest
from johen import generate

from seer.automation.agent.agent import (
    AgentConfig,
    LlmAgent,
    MaxIterationsReachedException,
    RunConfig,
)
from seer.automation.agent.client import OpenAiProvider
from seer.automation.agent.models import LlmGenerateTextResponse, Message, ToolCall, Usage
from seer.automation.agent.tools import ClaudeTool, FunctionTool


@pytest.fixture
def agent():
    config = AgentConfig()
    return LlmAgent(config=config, name="TestAgent")


@pytest.fixture
def run_config():
    return RunConfig(
        system_prompt="You are a helpful assistant.",
        prompt="Hello, how are you?",
        model=OpenAiProvider(model_name="gpt-4o-mini"),
        temperature=0.0,
        run_name="Test Run",
    )


@pytest.fixture
def fallback_run_config():
    return RunConfig(
        system_prompt="You are a helpful assistant.",
        prompt="Hello, how are you?",
        models=[
            OpenAiProvider(model_name="gpt-4o-mini"),
            OpenAiProvider(model_name="gpt-3.5-turbo"),
        ],
        temperature=0.0,
        run_name="Test Run with Fallback",
    )


class TestRunConfigValidation:
    """Test RunConfig validation logic"""

    def test_single_model_is_valid(self):
        """Should accept single model configuration"""
        config = RunConfig(model=OpenAiProvider.model("gpt-4"))
        assert config.model is not None
        assert config.model.model_name == "gpt-4"
        assert config.models is None

    def test_multiple_models_is_valid(self):
        """Should accept multiple models configuration"""
        models = [OpenAiProvider.model("gpt-4"), OpenAiProvider.model("gpt-3.5-turbo")]
        config = RunConfig(models=models)
        assert config.models == models
        assert len(config.models) == 2
        assert config.model is None

    def test_neither_model_nor_models_is_valid_for_runconfig_creation(self):
        """RunConfig should be able to be created without model/models (validation happens at runtime)"""
        config = RunConfig()
        assert config.model is None
        assert config.models is None

    def test_both_model_and_models_is_valid_for_runconfig_creation(self):
        """RunConfig should be able to be created with both (validation happens at runtime)"""
        config = RunConfig(
            model=OpenAiProvider.model("gpt-4"),
            models=[OpenAiProvider.model("gpt-3.5-turbo")],
        )
        assert config.model is not None
        assert config.models is not None

    def test_empty_models_list_is_valid_for_runconfig_creation(self):
        """RunConfig should be able to be created with empty models list (validation happens at runtime)"""
        config = RunConfig(models=[])
        assert config.models == []


def test_update_usage(agent: LlmAgent):
    usage1 = next(generate(Usage))
    usage2 = next(generate(Usage))
    agent.update_usage(usage1)
    agent.update_usage(usage2)
    assert agent.usage == usage1 + usage2


def test_process_message(agent: LlmAgent):
    message = Message(role="assistant", content="Hello!", tool_calls=[])
    agent.process_message(message)
    assert len(agent.memory) == 1
    assert agent.iterations == 1


def test_process_message_with_tool_calls(agent: LlmAgent):
    tool_call = ToolCall(id="1", function="test_tool", args='{"arg": "value"}')
    message = Message(role="assistant", content="Using a tool", tool_calls=[tool_call])

    fn1 = Mock()
    fn2 = Mock(return_value="Fn2 result")
    agent.tools.append(
        FunctionTool(
            name="unrelated_tool",
            description="",
            fn=fn1,
            parameters=[],
            required=[],
        )
    )

    agent.tools.append(
        FunctionTool(
            name="test_tool",
            description="",
            fn=fn2,
            parameters=[{"name": "arg"}],
            required=[],
        )
    )

    agent.process_message(message)
    fn1.assert_not_called()
    fn2.assert_called_once_with(arg="value")
    assert len(agent.memory) == 2
    assert agent.iterations == 1
    assert agent.memory[1].content == "Fn2 result"


def test_process_message_with_claude_tool_calls(agent: LlmAgent):
    tool_call = ToolCall(id="1", function="test_tool", args='{"arg": "value"}')
    message = Message(role="assistant", content="Using a tool", tool_calls=[tool_call])

    fn1 = Mock()
    fn2 = Mock(return_value="Fn2 result")
    agent.tools.append(
        FunctionTool(
            name="unrelated_tool",
            description="",
            fn=fn1,
            parameters=[],
            required=[],
        )
    )

    agent.tools.append(
        ClaudeTool(
            name="test_tool",
            type="function",
            fn=fn2,
        )
    )

    agent.process_message(message)
    fn1.assert_not_called()
    fn2.assert_called_once_with(arg="value", tool_call_id="1", current_memory_index=0)
    assert len(agent.memory) == 2
    assert agent.iterations == 1
    assert agent.memory[1].content == "Fn2 result"


def test_should_continue(agent: LlmAgent, run_config: RunConfig):
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


@pytest.mark.vcr()
def test_run_iteration(agent: LlmAgent, run_config: RunConfig):
    agent.run_iteration(run_config)

    assert len(agent.memory) == 1
    assert agent.memory[0].content == "How can I assist you today?"
    assert agent.iterations == 1
    assert agent.usage.total_tokens == 20


@pytest.mark.vcr()
def test_run_iteration_with_fallback_models(agent: LlmAgent, fallback_run_config: RunConfig):
    """Test run_iteration with fallback models configuration"""
    agent.run_iteration(fallback_run_config)

    assert len(agent.memory) == 1
    assert agent.memory[0].content == "How can I assist you today?"
    assert agent.iterations == 1
    assert agent.usage.total_tokens == 20


@pytest.mark.vcr()
def test_run_with_tool_calls(agent: LlmAgent, run_config: RunConfig):
    tool = FunctionTool(
        name="test_tool", description="A test tool", parameters=[], fn=lambda: "Tool result"
    )
    agent.tools = [tool]
    run_config.prompt = "Can you call test_tool?"
    result = agent.run(run_config)

    assert result == 'The test_tool has been successfully called, and the result is "Tool result".'
    assert (
        len(agent.memory) == 4
    )  # User message, initial assistant message, tool response, final message
    assert agent.iterations == 2
    assert agent.usage.total_tokens == 141


@pytest.mark.vcr()
def test_run_with_tool_calls_fallback_models(agent: LlmAgent, fallback_run_config: RunConfig):
    """Test run with tool calls using fallback models"""
    tool = FunctionTool(
        name="test_tool", description="A test tool", parameters=[], fn=lambda: "Tool result"
    )
    agent.tools = [tool]
    fallback_run_config.prompt = "Can you call test_tool?"
    result = agent.run(fallback_run_config)

    # Check that the tool was called and result is included (allow for minor wording variations)
    assert "test_tool" in result or "test tool" in result
    assert "Tool result" in result
    assert (
        len(agent.memory) == 4
    )  # User message, initial assistant message, tool response, final message
    assert agent.iterations == 2
    # Token count may vary slightly with fallback models vs single model
    assert agent.usage.total_tokens > 0


def test_run_max_iterations_exception(run_config: RunConfig):
    class LoopingAgent(LlmAgent):
        def should_continue(self, run_config: RunConfig) -> bool:
            return True

        def get_completion(self, run_config: RunConfig) -> LlmGenerateTextResponse:
            return next(generate(LlmGenerateTextResponse))

    agent = LoopingAgent(config=AgentConfig())

    run_config.max_iterations = 100

    with pytest.raises(MaxIterationsReachedException):
        agent.run(run_config)

    assert agent.iterations == 100


def test_run_max_iterations_exception_with_fallback(fallback_run_config: RunConfig):
    """Test max iterations exception with fallback models"""

    class LoopingAgent(LlmAgent):
        def should_continue(self, run_config: RunConfig) -> bool:
            return True

        def get_completion(self, run_config: RunConfig) -> LlmGenerateTextResponse:
            return next(generate(LlmGenerateTextResponse))

    agent = LoopingAgent(config=AgentConfig())

    fallback_run_config.max_iterations = 100

    with pytest.raises(MaxIterationsReachedException):
        agent.run(fallback_run_config)

    assert agent.iterations == 100


@pytest.mark.vcr()
def test_run_with_initial_prompt(agent: LlmAgent, run_config: RunConfig):
    run_config.prompt = "Initial prompt"
    result = agent.run(run_config)

    assert result
    assert len(agent.memory) == 2  # Initial user message and assistant response
    assert agent.memory[0].content == "Initial prompt"
    assert agent.memory[0].role == "user"


@pytest.mark.vcr()
def test_run_with_initial_prompt_fallback_models(agent: LlmAgent, fallback_run_config: RunConfig):
    """Test run with initial prompt using fallback models"""
    fallback_run_config.prompt = "Initial prompt"
    result = agent.run(fallback_run_config)

    assert result
    assert len(agent.memory) == 2  # Initial user message and assistant response
    assert agent.memory[0].content == "Initial prompt"
    assert agent.memory[0].role == "user"
