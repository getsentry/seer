import contextlib
import dataclasses
import datetime
from typing import Annotated, Iterator
from unittest import mock

from johen import generate
from johen.examples import Examples
from johen.generators import specialized

from seer.automation.agent.client import DummyGptClient, GptCompletionHandler
from seer.automation.models import SentryExceptionEntry, SentryFrame, StacktraceFrame
from seer.rpc import DummyRpcClient, RpcClientHandler

_now = datetime.datetime(2023, 1, 1)

Now = Annotated[datetime.datetime, Examples([_now])]
Past = Annotated[
    datetime.datetime, Examples(_now - delta for delta in specialized.positive_timedeltas if delta)
]
Future = Annotated[
    datetime.datetime, Examples(_now + delta for delta in specialized.positive_timedeltas if delta)
]

SentryFrameDict = Annotated[
    SentryFrame,
    Examples(
        (
            {**base_frame, **stacktrace_frame.model_dump(mode="json", by_alias=True)}
            for base_frame, stacktrace_frame in zip(
                generate(SentryFrame, generate_defaults="holes"),
                generate(StacktraceFrame, generate_defaults=False),
            )
        ),
    ),
]

InvalidEventEntry = Annotated[
    dict,
    Examples(
        ({"type": "not-a-valid-type", "data": {k: "hello"}} for k in specialized.printable_strings),
        ({"blah": v} for v in specialized.ints),
    ),
]

NoStacktraceExceptionEntry = Annotated[
    dict,
    Examples(
        (
            SentryExceptionEntry(
                type="exception",
                data={
                    "values": [
                        {
                            "type": "SomeError",
                            "value": "Yes im an error",
                            "stacktrace": {"frames": []},
                        }
                    ]
                },
            ).model_dump(mode="json"),
        ),
    ),
]


@dataclasses.dataclass
class RpcClientMock:
    client: DummyRpcClient
    mocked_path: str = "seer.automation.autofix.tasks.SentryRpcClient"

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
    mocked_path: str = "seer.automation.agent.agent.GptClient"

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
