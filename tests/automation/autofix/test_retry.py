"""
Tests that completions that fail during streaming are retried.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

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

ANTHROPIC_OVERLOADED_ERROR_DATA = {
    "type": "error",
    "error": {"type": "overloaded_error", "message": "Overloaded"},
}


class StreamFlaky:
    def __init__(
        self,
        stream_real: Iterator,
        retryable_exception: Exception,
        num_chunks_before_erroring: int = 2,
    ):
        self._iterator = stream_real
        self._retryable_exception = retryable_exception
        self._num_chunks_before_erroring = num_chunks_before_erroring
        self.response = getattr(stream_real, "response", None)
        # Need this attribute for Anthropic's stream

    def __iter__(self):
        for num_chunks_generated, chunk in enumerate(self._iterator):
            if num_chunks_generated == self._num_chunks_before_erroring:
                raise self._retryable_exception
            yield chunk


_T = TypeVar("_T", bound=LlmProvider)


def flakify(
    provider_class: type[_T],
    retryable_exception: Exception,
    get_obj_with_create_stream_method_from_client: Callable,
    create_stream_method_name: str,
) -> type[_T]:
    """
    Mock an LLM provider that will return a flaky stream for the first `max_num_flaky_clients` clients.
    """

    def flakify(create_stream: Callable[[Any], Iterator]):

        @wraps(create_stream)
        def wrapped(*args, **kwargs):
            stream = create_stream(*args, **kwargs)
            return StreamFlaky(stream, retryable_exception)

        return wrapped

    @dataclass
    class FlakyProvider(provider_class):
        num_flaky_clients: int = 0
        max_num_flaky_clients: int = 2

        def get_client(self):
            client = provider_class.get_client()
            if self.num_flaky_clients < self.max_num_flaky_clients:
                self.num_flaky_clients += 1
                # Put the client in a unhealthy state, i.e., make all of its streams flaky
                obj_with_create_stream_method = get_obj_with_create_stream_method_from_client(
                    client
                )
                create_stream_method = getattr(
                    obj_with_create_stream_method, create_stream_method_name
                )
                setattr(
                    obj_with_create_stream_method,
                    create_stream_method_name,
                    flakify(create_stream_method),
                )
            return client

    return FlakyProvider


# Define your flaky providers here
AnthropicProviderFlaky = flakify(
    AnthropicProvider,
    retryable_exception=anthropic.APIStatusError(
        message=str(ANTHROPIC_OVERLOADED_ERROR_DATA),
        response=httpx.Response(
            status_code=529, request=httpx.Request("POST", "https://mocked.com")
        ),  # https://docs.anthropic.com/en/api/errors#http-errors
        body=ANTHROPIC_OVERLOADED_ERROR_DATA,
    ),
    get_obj_with_create_stream_method_from_client=lambda client: client.messages,
    create_stream_method_name="create",
)


# Give this function a name instead of making it a lambda so that the cassette is named
def create_flaky_anthropic():
    return AnthropicProviderFlaky.model("claude-3-5-sonnet@20240620")


@pytest.fixture(
    scope="module",
    params=[
        create_flaky_anthropic,
    ],
)
def create_flaky_provider(request: pytest.FixtureRequest) -> Callable[[], LlmProvider]:
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
def run_config(create_flaky_provider: Callable[[], LlmProvider]):
    return RunConfig(
        system_prompt="You are a helpful assistant for fixing code.",
        prompt="Fix this bug.",
        model=create_flaky_provider(),
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
            f"Unexpected exception type: {type(exception_info.value)}. Handle this in an elif block above."
        )


@pytest.mark.vcr()
def test_provider_without_exception_indicator(autofix_agent: AutofixAgent, run_config: RunConfig):
    """
    Test behavior when a provider doesn't implement `is_completion_exception_retryable`.
    """
    if not isinstance(run_config.model, AnthropicProvider):
        pytest.skip("This test was already ran for Anthropic. Skipping to avoid redundancy.")

    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]

    @contextmanager
    def temp_delete_static_method(cls: type, method_name: str):
        try:
            method = getattr(cls, method_name)
            delattr(cls, method_name)
            yield
        finally:
            setattr(cls, method_name, staticmethod(method))

    # Test that if the API is unhealthy and is_completion_exception_retryable isn't implemented, the overload error is raised
    with temp_delete_static_method(AnthropicProvider, "is_completion_exception_retryable"):
        assert not hasattr(run_config.model, "is_completion_exception_retryable")
        with pytest.raises(Exception) as exception_info:
            _ = autofix_agent.get_completion(run_config)
    assert hasattr(run_config.model, "is_completion_exception_retryable")
    assert run_config.model.is_completion_exception_retryable(exception_info.value)

    # Test that if the API is healthy and is_completion_exception_retryable isn't implemented, the completion is returned
    run_config.model = AnthropicProvider.model("claude-3-5-sonnet@20240620")  # healthy API
    with temp_delete_static_method(AnthropicProvider, "is_completion_exception_retryable"):
        assert not hasattr(run_config.model, "is_completion_exception_retryable")
        response = autofix_agent.get_completion(run_config)
    assert EXPECTED_MESSAGE in response.message.content


@pytest.mark.vcr()
def test_retrying_succeeds(autofix_agent: AutofixAgent, run_config: RunConfig):
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
    assert isinstance(run_config.model, AnthropicProviderFlaky)
    assert run_config.model.num_flaky_clients < run_config.model.max_num_flaky_clients
    # Sanity check that the API will indeed be flaky
    response = autofix_agent.get_completion(run_config)
    assert EXPECTED_MESSAGE in response.message.content
