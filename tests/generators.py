import datetime
from typing import Annotated

from seer import generator
from seer.automation.autofix.models import SentryExceptionEntry, SentryFrame
from seer.generator import Examples

_now = datetime.datetime(2023, 1, 1)

Now = Annotated[datetime.datetime, Examples([_now])]
Past = Annotated[
    datetime.datetime, Examples(_now - delta for delta in generator.positive_timedeltas if delta)
]
Future = Annotated[
    datetime.datetime, Examples(_now + delta for delta in generator.positive_timedeltas if delta)
]

NarrowSentryFrameWithFilename = Annotated[
    SentryFrame,
    Examples(
        (
            [
                {
                    **base_frame,
                    filename: filename,
                }
                for base_frame, filename, r in zip(
                    generator.generate(SentryFrame), generator.file_names, generator.gen
                )
            ]
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
