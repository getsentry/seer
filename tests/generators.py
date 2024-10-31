import datetime
from typing import Annotated

from johen import generate
from johen.examples import Examples
from johen.generators import specialized

from seer.automation.models import SentryExceptionEntry, SentryFrame, StacktraceFrame

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
