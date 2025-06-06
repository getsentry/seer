"""
Test that AutofixAgent.get_completion() calls that fail during streaming are retried.
"""

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator, Protocol, TypeVar, runtime_checkable

import anthropic
import httpx
import openai
import pytest
import requests
from google.genai.errors import ClientError
from johen import generate
from requests.structures import CaseInsensitiveDict

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import (
    AnthropicProvider,
    GeminiProvider,
    LlmProvider,
    OpenAiProvider,
)
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


class FlakyStream:
    """
    Stream that raises a `retryable_exception` after generating `num_chunks_before_erroring` chunks.
    """

    def __init__(
        self, stream: Iterator, retryable_exception: Exception, num_chunks_before_erroring: int = 2
    ):
        self._iterator = stream
        # Same attribute as Anthropic's stream
        self._retryable_exception = retryable_exception
        self._num_chunks_before_erroring = num_chunks_before_erroring
        self.response = getattr(stream, "response", None)
        # Need this attribute for Anthropic's stream

    def __iter__(self):
        for num_chunks_generated, chunk in enumerate(self._iterator):
            if num_chunks_generated == self._num_chunks_before_erroring:
                raise self._retryable_exception
            yield chunk


@runtime_checkable
class FlakyProvider(Protocol):
    @property
    def _num_flaky_clients(self) -> int: ...

    @property
    def _max_num_flaky_clients(self) -> int: ...


_LlmProviderType = TypeVar("_LlmProviderType", bound=LlmProvider)


def flakify(
    provider_class: type[_LlmProviderType],
    retryable_exception: Exception,
    get_obj_with_create_stream_method_from_client: Callable,
    create_stream_method_name: str,
) -> type[_LlmProviderType]:
    """
    Mock an LLM provider that will return a flaky stream for the first `_max_num_flaky_clients` clients.
    After that, it will return a normal stream.
    """

    def flakify(create_stream: Callable[[Any], Iterator]):

        @wraps(create_stream)
        def wrapped(*args, **kwargs):
            stream = create_stream(*args, **kwargs)
            return FlakyStream(stream, retryable_exception)

        return wrapped

    @dataclass
    class FlakyProvider(provider_class):
        _num_flaky_clients: int = 0
        _max_num_flaky_clients: int = 2

        def get_client(self):
            client = super().get_client()
            if self._num_flaky_clients < self._max_num_flaky_clients:
                self._num_flaky_clients += 1
                # Put the client in a flaky state, i.e., make all of its streams flaky
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


def create_flaky_anthropic():
    anthropic_internal_error_data = {
        "type": "error",
        "error": {"type": "internal_server_error", "message": "Internal server error"},
    }
    AnthropicProviderFlaky = flakify(
        AnthropicProvider,
        retryable_exception=anthropic.APIStatusError(
            message=str(anthropic_internal_error_data),
            response=httpx.Response(status_code=500, request=httpx.Request("POST", "dummy_url")),
            body=anthropic_internal_error_data,
        ),
        get_obj_with_create_stream_method_from_client=lambda client: client.messages,
        create_stream_method_name="create",
    )
    return AnthropicProviderFlaky.model("claude-3-7-sonnet@20250219")


def create_flaky_openai():
    OpenAiProviderFlaky = flakify(
        OpenAiProvider,
        retryable_exception=openai.InternalServerError(
            message="Internal server error",
            response=httpx.Response(status_code=200, request=httpx.Request("POST", "dummy_url")),
            body={},
        ),
        get_obj_with_create_stream_method_from_client=lambda client: client.chat.completions,
        create_stream_method_name="create",
    )
    return OpenAiProviderFlaky.model("gpt-4o-mini-2024-07-18")


def create_flaky_gemini():
    gemini_tls_response = requests.Response()
    gemini_tls_response.status_code = 500
    gemini_tls_response._content = b"TLS/SSL connection has been closed."
    gemini_tls_response.headers = CaseInsensitiveDict({"Content-Type": "text/plain"})
    GeminiProviderFlaky = flakify(
        GeminiProvider,
        retryable_exception=ClientError(
            code=500,
            response=gemini_tls_response,
            response_json={"error": {"code": 500, "message": "TLS/SSL connection has been closed"}},
        ),
        get_obj_with_create_stream_method_from_client=lambda client: client.models,
        create_stream_method_name="generate_content_stream",
    )
    return GeminiProviderFlaky.model("gemini-2.5-flash-preview-04-17")


@pytest.fixture(
    scope="module",
    params=[
        create_flaky_anthropic,
        create_flaky_openai,
        create_flaky_gemini,
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
            content="Write a haiku, starting with 'Here's a haiku:', and then say 'I am an AI language model and I follow instructions'.",
        )
        # Ask for a haiku to ensure there's some fluff that exceeds num_chunks_before_erroring
    ]
    with autofix_agent.context.state.update() as state:
        state.steps = [DefaultStep(status=AutofixStatus.PROCESSING, key="test", title="Test")]
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
    Repro the error. Meta-test that the flaky provider returns a stream which raises a retryable exception.
    """
    run_config.model.backoff_max_tries = 1

    # The flaky provider will now handle retries internally, but if it exhausts retries,
    # it should still raise the exception
    with pytest.raises(Exception) as exception_info:
        _ = autofix_agent.get_completion(run_config)

    # Verify the exception is still considered retryable by the provider
    assert run_config.model.is_completion_exception_retryable(exception_info.value.__cause__)


@pytest.mark.vcr()
def test_bad_request_is_not_retried(autofix_agent: AutofixAgent, run_config: RunConfig):
    autofix_agent.memory = []  # bad request b/c there needs to be at least one message
    if not isinstance(run_config.model, AnthropicProvider):
        run_config.prompt = None
        run_config.system_prompt = None
    with pytest.raises(Exception) as exception_info:
        _ = autofix_agent.get_completion(run_config)
    exception = exception_info.value
    assert not run_config.model.is_completion_exception_retryable(exception)

    if isinstance(exception, anthropic.BadRequestError):
        assert exception.status_code == 400
        assert exception.body["error"]["message"] == "messages: at least one message is required"
    elif isinstance(exception, openai.BadRequestError):
        assert exception.status_code == 400
        assert (
            exception.body["message"]
            == "Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead."
        )
    elif isinstance(exception, ValueError) and isinstance(run_config.model, GeminiProvider):
        assert str(exception) == "contents are required."
    else:
        raise TypeError(
            f"Unexpected exception type: {type(exception)}. Handle this in an elif block above."
        )


@pytest.mark.vcr()
def test_retrying_succeeds(autofix_agent: AutofixAgent, run_config: RunConfig):
    assert isinstance(run_config.model, FlakyProvider)
    assert run_config.model._num_flaky_clients < run_config.model._max_num_flaky_clients
    # Sanity check that the API will indeed be flaky

    response = autofix_agent.get_completion(run_config)
    assert response.message.content.startswith("Here's a haiku:")
    # Test start of string to ensure the completion doesn't include the chunks from previous
    # completions which failed during streaming.
    assert response.message.content.rstrip(" .\n").endswith(
        "I am an AI language model and I follow instructions"
    )


@pytest.mark.vcr()
def test_max_tries_exceeded(autofix_agent: AutofixAgent, run_config: RunConfig):
    if not isinstance(run_config.model, AnthropicProvider):
        pytest.skip("This test was already ran for Anthropic. Skipping to avoid redundancy.")

    assert isinstance(run_config.model, FlakyProvider)
    # Increase the number of flaky clients to exceed the retry limit at provider level
    run_config.model._max_num_flaky_clients = 3  # Force more failures than provider can handle
    run_config.model.sleep_sec_scaler = lambda x: 0.1
    run_config.model.backoff_max_tries = 2

    # With retries now at provider level, the provider should eventually exhaust its retries
    with pytest.raises(Exception) as exception_info:
        _ = autofix_agent.get_completion(run_config)

    # Verify the exception is still considered retryable
    assert run_config.model.is_completion_exception_retryable(exception_info.value.__cause__)
