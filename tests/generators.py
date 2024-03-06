import datetime
from typing import Annotated

from seer.generator import Examples, positive_timedeltas

_now = datetime.datetime(2023, 1, 1)

Now = Annotated[datetime.datetime, Examples([_now])]
Past = Annotated[
    datetime.datetime, Examples(_now - delta for delta in positive_timedeltas if delta)
]
Future = Annotated[
    datetime.datetime, Examples(_now + delta for delta in positive_timedeltas if delta)
]
