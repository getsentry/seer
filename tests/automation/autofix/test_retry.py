"""
Tests that completions that fail during streaming are retried.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator

import anthropic
import httpx
import pytest
from johen import generate

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmProvider
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    DefaultStep,
)
from seer.automation.state import LocalMemoryState

EXPECTED_MESSAGE = "I am an AI language model and I follow instructions"


class StreamFlaky:
    def __init__(
        self,
        stream_real: Iterator,
        retryable_exception: Exception,
        num_chunks_before_erroring: int = 2,
    ):
        self._iterator = stream_real
        self.retryable_exception = retryable_exception
        self.num_chunks_before_erroring = num_chunks_before_erroring
        self.response = getattr(stream_real, "response", None)

    def __iter__(self):
        for num_chunks_generated, chunk in enumerate(self._iterator):
            if num_chunks_generated == self.num_chunks_before_erroring:
                raise self.retryable_exception
            yield chunk


def flakify(create: Callable[[Any], Iterator], retryable_exception: Exception):

    @wraps(create)
    def wrapped(*args, **kwargs):
        stream_real = create(*args, **kwargs)
        return StreamFlaky(stream_real, retryable_exception)

    return wrapped


@dataclass
class AnthropicProviderFlaky(AnthropicProvider):
    """
    A provider that returns a flaky client for the first `max_num_flaky_clients`.

    A flaky client raises a retryable exception after generating some chunks.
    """

    num_flaky_clients: int = 0
    max_num_flaky_clients: int = 2

    def get_client(self) -> anthropic.AnthropicVertex:
        client = AnthropicProvider.get_client()
        if self.num_flaky_clients < self.max_num_flaky_clients:
            # The API is in an overloaded state
            overloaded_error_data = {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            }
            retryable_exception = anthropic.APIStatusError(
                message=str(overloaded_error_data),
                response=httpx.Response(
                    status_code=529, request=httpx.Request("POST", "https://moc.ked")
                ),  # https://docs.anthropic.com/en/api/errors#http-errors
                body=overloaded_error_data,
            )
            self.num_flaky_clients += 1
            client.messages.create = flakify(client.messages.create, retryable_exception)
        return client


@pytest.fixture(
    scope="module",
    params=[
        AnthropicProviderFlaky.model("claude-3-5-sonnet@20240620"),
    ],
)
def flaky_provider(request: pytest.FixtureRequest) -> LlmProvider:
    return request.param


@pytest.fixture
def context():
    request = next(generate(AutofixRequest))
    continuation = AutofixContinuation(request=request)
    state = LocalMemoryState(val=continuation)
    return AutofixContext(state=state, event_manager=AutofixEventManager(state=state))


@pytest.fixture
def autofix_agent(context: AutofixContext):
    config = AgentConfig()
    autofix_agent = AutofixAgent(config=config, context=context, name="TestAutofixAgent")
    autofix_agent.memory = [
        Message(
            role="user",
            content=f"Write a haiku and then say '{EXPECTED_MESSAGE}'.",
        )
        # Ask for a haiku to ensure there's some fluff that exceeds num_chunks_before_erroring
    ]
    return autofix_agent


@pytest.fixture
def run_config(flaky_provider: LlmProvider):
    return RunConfig(
        system_prompt="You are a helpful assistant for fixing code.",
        prompt="Fix this bug.",
        model=flaky_provider,
        temperature=0.0,
        run_name="Test Autofix Run",
    )


@pytest.mark.vcr()
def test_flaky_stream(autofix_agent: AutofixAgent, run_config: RunConfig):
    """
    Meta-test that the flaky provider returns a stream which raises a retryable exception.
    """
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    with pytest.raises(Exception) as exception_info:
        _ = autofix_agent.get_completion(run_config, max_tries=1)
    assert run_config.model.is_completion_exception_retryable(exception_info.value)


@pytest.mark.vcr()
def test_bad_request_is_not_retried(autofix_agent: AutofixAgent, run_config: RunConfig):
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    autofix_agent.memory = []
    with pytest.raises(Exception) as exception_info:
        _ = autofix_agent.get_completion(run_config)
    assert not run_config.model.is_completion_exception_retryable(exception_info.value)
    if isinstance(exception_info.value, anthropic.BadRequestError):
        assert exception_info.value.status_code == 400
        assert (
            exception_info.value.body["error"]["message"]
            == "messages: at least one message is required"
        )
    else:
        raise Exception(
            f"Unexpected exception type: {type(exception_info.value)}. Handle this here."
        )


@pytest.mark.vcr()
def test_provider_without_exception_indicator(autofix_agent: AutofixAgent, run_config: RunConfig):
    """
    Test that it's ok if a provider doesn't implement `is_completion_exception_retryable` and the corresponding API is healthy.
    """
    if not isinstance(run_config.model, AnthropicProvider):
        pytest.skip("This test was already ran for Anthropic. Skipping to avoid redundancy.")

    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]

    @contextmanager
    def temp_delete_static_method(obj: Any, method_name: str):
        try:
            method = getattr(obj, method_name)
            delattr(obj, method_name)
            yield
        finally:
            setattr(obj, method_name, staticmethod(method))

    run_config.model = AnthropicProvider.model("claude-3-5-sonnet@20240620")  # healthy API
    with temp_delete_static_method(type(run_config.model), "is_completion_exception_retryable"):
        response = autofix_agent.get_completion(run_config)
        assert EXPECTED_MESSAGE in response.message.content


@pytest.mark.vcr()
def test_retrying_succeeds(autofix_agent: AutofixAgent, run_config: RunConfig):
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    run_config.model.is_completion_exception_retryable
    response = autofix_agent.get_completion(run_config)
    assert EXPECTED_MESSAGE in response.message.content
