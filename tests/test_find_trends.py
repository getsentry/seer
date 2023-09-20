import random
from datetime import datetime
from typing import Optional

import pytest

from seer.trend_detection.trend_detector import find_trends

random.seed(a=0, version=2)

hour = 3600

@pytest.fixture
def timeseries():
    n = 24 * 7
    return [
        p95 + random.gauss(mu=0.0, sigma=5_000_000)
        for p95 in ([50_000_000] * (24 * 90 - n) + [150_000_000] * n)
    ]


def ema(val1: Optional[float], val2: float, weight: float):
    if val1 is None:
        return val2
    return val1 * (1 - weight) + val2 * weight


def format_series(
    series,
    start=1694023200,  # doesnt really matter just need a timestamp
):
    return [
        (start + i * hour, [{"count": val}])
        for i, val in enumerate(series)
    ]


def check_trends(series):
    series = format_series(series)
    sort_function = "-trend_percentage()"
    zerofilled = True
    data = {
        "1,1": {
            "data": series,
            "data_start": series[0][0],
            "data_end": series[-1][0] + hour,
            # "request_start": p95s[-1][0] - 1 * 24 * hour,  # last 24 hours
            "request_start": series[0][0],  # whole 14 days
            "request_end": series[-1][0] + hour,
        },
    }

    return find_trends(data, sort_function, zerofilled, False)


def test_foo(timeseries):
    short_weight = 2 / 21
    long_weight = 2 / 41
    threshold = 0.1

    s: Optional[float] = None
    l: Optional[float] = None

    ema_breakpoints = []

    for i, val in enumerate(timeseries):
        try:
            rel_old = (s - l) / abs(l)
        except (TypeError, ZeroDivisionError):
            rel_old = None

        s = ema(s, val, short_weight)
        l = ema(l, val, long_weight)

        try:
            rel_new = (s - l) / abs(l)
        except (TypeError, ZeroDivisionError):
            rel_new = None

        if rel_old is None or rel_new is None:
            continue

        if (
            i >= 6
            and rel_old < threshold
            and rel_new > threshold
        ):
            ema_breakpoints.append(i)

    for i in ema_breakpoints:
        count = 14 * 24
        delta = 27
        series = timeseries[i + delta - count: i + delta]
        print(check_trends(series))

    assert 0, ema_breakpoints
