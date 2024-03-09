import datetime
from typing import Annotated

from seer import generator
from seer.automation.autofix.models import SentryExceptionEntry, SentryFrame, StacktraceFrame
from seer.generator import Examples

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
            [
                {**base_frame, **stacktrace_frame.model_dump(mode="json")}
                for base_frame, stacktrace_frame in zip(
                    generator.generate(SentryFrame, include_defaults="holes"),
                    generator.generate(StacktraceFrame, include_defaults="holes"),
                )
            ]
            for r in generator.gen
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
