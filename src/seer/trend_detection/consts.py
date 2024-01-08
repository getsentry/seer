"""
This module contains :class:`TimeSeriesChangePoint`

:class:`TimeSeriesChangePoint` is the return type of many of the Kats detection
algorithms.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

FigSize = Tuple[int, int]


def _log_error(msg: str) -> ValueError:
    logging.error(msg)
    return ValueError(msg)


class TimeSeriesChangePoint:
    """Object returned by detector classes.

    Attributes:

        start_time: Start time of the change.
        end_time: End time of the change.
        confidence: The confidence of the change point.
    """

    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        confidence: float,
    ) -> None:
        self._start_time = start_time
        self._end_time = end_time
        self._confidence = confidence

    @property
    def start_time(self) -> pd.Timestamp:
        return self._start_time

    @property
    def end_time(self) -> pd.Timestamp:
        return self._end_time

    @property
    def confidence(self) -> float:
        return self._confidence

    def __repr__(self) -> str:
        return (
            f"TimeSeriesChangePoint(start_time: {self._start_time}, end_time: "
            f"{self._end_time}, confidence: {self._confidence})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TimeSeriesChangePoint)
            and (self._start_time == other._start_time)
            and (self._end_time == other._end_time)
            and (self._confidence == other._confidence)
        )

    def __hash__(self) -> int:
        # Allow subclasses to override __repr__ without affecting __hash__.
        return hash(f"{self._start_time},{self._end_time},{self._confidence}")


__all__ = ["TimeSeriesChangePoint"]
