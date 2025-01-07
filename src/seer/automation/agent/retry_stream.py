"""
Automatically retry a streamed completion request if the original request is
interrupted by an error. The retried request will pick up from where the
previous one left off, i.e., its chunks are streamed and saved.
"""

import logging
import time
from dataclasses import dataclass, field
from math import ceil
from typing import Callable, cast

from seer.automation.agent.models import Message, ToolCall, Usage
from seer.automation.agent.tools import FunctionTool

logger = logging.getLogger(__name__)


SleepSecScaler = Callable[[int], float | int]
"""
Inputs the current retry number (starting from 1).
Outputs the number of seconds to sleep.
"""


def exponential(retries: int) -> int:
    return 2**retries


class MaxRetriesExceededDuringStreamError(Exception):
    """
    Raised when we've retried too many times.
    """


class Backoff:
    def __init__(
        self,
        max_retries: int = 8,
        sleep_sec_scaler: SleepSecScaler = exponential,
    ):
        self.max_retries = max_retries
        self.sleep_sec_scaler = sleep_sec_scaler
        self._retries = 0

    def __call__(self, from_exception: Exception | None = None):
        if self._retries >= self.max_retries:
            raise MaxRetriesExceededDuringStreamError(
                f"Tried all {self.max_retries} retries. Not retrying anymore."
            ) from from_exception

        self._retries += 1
        sleep_sec = self.sleep_sec_scaler(self._retries)
        logger.info(
            f"Sleeping for {sleep_sec} seconds before attempting "
            f"retry {self._retries} out of {self.max_retries}."
        )
        time.sleep(sleep_sec)


@dataclass
class PartialCompletion:
    """
    The state of a completion constructed from a stream.
    """

    content_chunks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    completion_tokens_approx: int = 0
    avg_num_chars_per_token: int = 5
    # The rule of thumb is 4, but tokenizer vocabs have blown up since then

    def update(self, chunk: str | ToolCall | Usage) -> None:
        if isinstance(chunk, str):
            self.content_chunks.append(chunk)
            num_tokens_approx = ceil(len(chunk) / self.avg_num_chars_per_token)
            # A content chunk isn't a single token as of a few months ago,
            # and the event doesn't indicate how many tokens are in the chunk.
            # So there isn't a clean and quick way to exactly calculate the
            # number of tokens in this chunk. I think an approximation is fine
            # for our purposes.
            self.completion_tokens_approx += num_tokens_approx
        elif isinstance(chunk, ToolCall):
            self.tool_calls.append(chunk)
        elif isinstance(chunk, Usage):
            pass
            # The final Usage obj is not accurate for a partial completion.
            # It's only accurate if the stream reached a
            # RawMessageDeltaEvent (chunk.type == "message_delta").
            # That event is only reached at the end of a block of
            # text or a tool call, which likely didn't happen when the
            # stream was interrupted by an overload_error.
            # https://docs.anthropic.com/en/api/messages-streaming
        else:
            raise TypeError(f"Got an unexpected type of chunk from streaming: {type(chunk)}")

    def __bool__(self) -> bool:
        return bool(self.content_chunks) or bool(self.tool_calls)


def _generate_text_stream_retry_recursive(
    *,
    does_exception_indicate_retry: Callable[[Exception], bool],
    backoff: Backoff,
    model,
    messages: list[Message],
    system_prompt: str | None,
    tools: list[FunctionTool] | None,
    temperature: float | None,
    max_tokens: int,
    timeout: float | None,
):
    from seer.automation.agent.client import LlmProvider

    model = cast(LlmProvider, model)

    stream = model.generate_text_stream(
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    partial_completion = PartialCompletion()

    try:
        for chunk in stream:
            partial_completion.update(chunk)
            yield chunk
    except Exception as exception:
        if not does_exception_indicate_retry(exception):
            raise exception

        if partial_completion.content_chunks:
            logger.info(
                "Last string chunk from this stream: "
                f"{repr(partial_completion.content_chunks[-1])}"
            )

        backoff(from_exception=exception)

        if partial_completion:
            partial_message = model.construct_message_from_stream(
                content_chunks=partial_completion.content_chunks,
                tool_calls=partial_completion.tool_calls,
            )
            messages = messages + [partial_message]
            # The Anthropic API will resume generating this last (assistant)
            # message.
            if messages[-1].content:
                messages[-1].content = messages[-1].content.rstrip()
                # Hack to avoid:
                # BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'messages: final assistant content cannot end with trailing whitespace'}}

        max_tokens -= partial_completion.completion_tokens_approx

        stream_continued = _generate_text_stream_retry_recursive(
            does_exception_indicate_retry=does_exception_indicate_retry,
            backoff=backoff,
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        yield from stream_continued


def generate_text_stream_retry(
    *,
    does_exception_indicate_retry: Callable[[Exception], bool],
    model,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    system_prompt: str | None = None,
    tools: list[FunctionTool] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries_during_stream: int | None = None,
    sleep_sec_scaler: SleepSecScaler | None = None,
):
    """
    Like `model.generate_text_stream` except it retries during a stream if, during
    streaming, an exception is raised for which
    `does_exception_indicate_retry(exception)` is True.
    """
    if messages is None:
        messages = []
    if prompt:
        messages = messages + [Message(role="user", content=prompt)]
        # Choosing to not append to messages to avoid weird state
    # NOTE: construct initial messages instead of passing them through. If prompt is
    # passed through every call, then the messages submitted to the API after a retry
    # will be ordered like—
    # messages, partial completion messages, prompt message
    # —instead of—
    # messages, prompt messsage, partial completion messages

    if not max_tokens:
        max_tokens = 8192
    if max_retries_during_stream is None:
        max_retries_during_stream = 8
    if sleep_sec_scaler is None:
        sleep_sec_scaler = exponential

    backoff = Backoff(max_retries_during_stream, sleep_sec_scaler)

    yield from _generate_text_stream_retry_recursive(
        does_exception_indicate_retry=does_exception_indicate_retry,
        backoff=backoff,
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
