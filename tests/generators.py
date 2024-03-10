import contextlib
import dataclasses
import datetime
from typing import Annotated, Iterator
from unittest import mock

from seer import generator
from seer.automation.agent.client import DummyGptClient, GptCompletionHandler
from seer.automation.autofix.models import SentryExceptionEntry, SentryFrame, StacktraceFrame
from seer.generator import Examples
from seer.rpc import DummyRpcClient, RpcClientHandler

_now = datetime.datetime(2023, 1, 1)

Now = Annotated[datetime.datetime, Examples([_now])]
Past = Annotated[
    datetime.datetime, Examples(_now - delta for delta in generator.positive_timedeltas if delta)
]
Future = Annotated[
    datetime.datetime, Examples(_now + delta for delta in generator.positive_timedeltas if delta)
]

SentryFrameDict = Annotated[
    SentryFrame,
    Examples(
        (
            {**base_frame, **stacktrace_frame.model_dump(mode="json", by_alias=True)}
            for base_frame, stacktrace_frame in zip(
                generator.generate(SentryFrame, include_defaults="holes"),
                generator.generate(StacktraceFrame, include_defaults=False),
            )
        ),
    ),
]

InvalidEventEntry = Annotated[
    dict,
    Examples(
        ({"type": "not-a-valid-type", "data": {k: "hello"}} for k in generator.printable_strings),
        ({"blah": v} for v in generator.ints),
    ),
]

NoStacktraceExceptionEntry = Annotated[
    dict,
    Examples(
        (
            SentryExceptionEntry(type="exception", data={"values": []}).model_dump(mode="json")
            for _ in generator.gen
        ),
        (
            SentryExceptionEntry(
                type="exception", data={"values": [{"stacktrace": {"frames": []}}]}
            ).model_dump(mode="json"),
        ),
    ),
]


@dataclasses.dataclass
class RpcClientMock:
    client: DummyRpcClient
    mocked_path: str = "seer.rpc.SentryRpcClient"

    def _enabled(self, **handlers: RpcClientHandler) -> Iterator[DummyRpcClient]:
        old_handlers = self.client.handlers
        with mock.patch(self.mocked_path) as target:
            target.return_value = self.client
            self.client.handlers = {**old_handlers, **handlers}
            try:
                yield self.client
            finally:
                self.client.handlers = old_handlers

    enabled = contextlib.contextmanager(_enabled)


@dataclasses.dataclass
class GptClientMock:
    client: DummyGptClient
    mocked_path: str = "seer.automation.agent.client.GptClient"

    @contextlib.contextmanager
    def enabled(self, *handlers: GptCompletionHandler) -> Iterator[DummyGptClient]:
        old_handlers = self.client.handlers
        with mock.patch(self.mocked_path) as target:
            target.return_value = self.client
            self.client.handlers = [*old_handlers, *handlers]
            try:
                yield self.client
            finally:
                self.client.handlers = old_handlers
