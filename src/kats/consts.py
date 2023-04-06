# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains some of the key data structures in the Kats library,
including :class:`TimeSeriesData`, :class:`TimeSeriesChangePoint`, and
:class:`TimeSeriesIterator`.

:class:`TimeSeriesChangePoint` is the return type of many of the Kats detection
algorithms.

:class:`TimeSeriesData` is the fundamental data structure in the Kats library,
that gives uses access to a host of forecasting, detection, and utility
algorithms right at the user's fingertips.
"""

from __future__ import annotations

import builtins
import copy
import datetime
import logging
from collections.abc import Iterable
from enum import auto, Enum, unique
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_numeric_dtype
from pandas.tseries.frequencies import to_offset

FigSize = Tuple[int, int]


# Constants
DEFAULT_TIME_NAME = "time"  # Default name for the time column in TimeSeriesData
DEFAULT_VALUE_NAME = "value"  # Default name for the value column in TimeSeriesData
PREFIX_OP_1 = "_kats.1"  # Internal prefix used when merging two TimeSeriesData objects
PREFIX_OP_2 = (
    "_kats.2"  # Second internal prefix used when merging two TimeSeriesData objects
)
INTERPOLATION_METHODS = {
    "linear",
    "bfill",
    "ffill",
}  # List of possible interpolation methods

IRREGULAR_GRANULARITY_ERROR = (
    "This algorithm or this parameter setup does not support input data with irregular data granularity. "
    "Please update your query to ensure that your data have fixed granularity."
)


class KatsError(Exception):
    pass


class DataError(KatsError):
    pass


class DataIrregualarGranularityError(DataError):
    pass


class DataInsufficientError(DataError):
    pass


class ParameterError(KatsError):
    pass


class InternalError(KatsError):
    pass


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
        return hash("{self._start_time},{self._end_time},{self._confidence}")

class Params:
    def __init__(self) -> None:
        pass

    def validate_params(self) -> None:
        pass


class IntervalAnomaly:
    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        if start >= end:
            raise ValueError("Start value is supposed to be larger than end value.")
        self.start: pd.Timestamp = start
        self.end: pd.Timestamp = end

    @property
    def second_len(self) -> int:
        return (self.end - self.start) / np.timedelta64(1, "s")


@unique
class ModelEnum(Enum):
    """
    This enum lists the options of models to be set for default search space in
    hyper-parameter tuning.
    """

    ARIMA = auto()
    SARIMA = auto()
    PROPHET = auto()
    HOLTWINTERS = auto()
    LINEAR = auto()
    QUADRATIC = auto()


@unique
class SearchMethodEnum(Enum):
    """
    This enum lists the options of search algorithms to be used in
    hyper-parameter tuning.
    """

    GRID_SEARCH = auto()
    RANDOM_SEARCH_UNIFORM = auto()
    RANDOM_SEARCH_SOBOL = auto()
    BAYES_OPT = auto()


@unique
class OperationsEnum(Enum):
    """
    This enum lists all the mathematical operations that can be performed on
    :class:`TimeSeriesData` objects.
    """

    ADD = auto()
    SUB = auto()
    DIV = auto()
    MUL = auto()


__all__ = [
    "ModelEnum",
    "OperationsEnum",
    "Params",
    "SearchMethodEnum",
    "TimeSeriesChangePoint",
    "TimeSeriesData",
    "TimeSeriesIterator",
    "TSIterator",
]
