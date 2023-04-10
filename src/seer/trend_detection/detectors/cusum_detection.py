# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
CUSUM stands for cumulative sum, it is a changepoint detection algorithm.

It has two main components:

  1. Locate the change point: The algorithm iteratively estimates the means
      before and after the change point and finds the change point
      maximizing/minimizing the cusum value until the change point has
      converged. The starting point for the change point is at the middle.

  2. Hypothesis testing: Conducting log likelihood ratio test where the null
      hypothesis has no change point with one mean and the alternative
      hypothesis has a change point with two means.

"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seer.trend_detection.consts import TimeSeriesChangePoint
from scipy.stats import chi2

pd.options.plotting.matplotlib.register_converters = True

_log: logging.Logger = logging.getLogger("cusum_detection")


@dataclass
class CUSUMDefaultArgs:
    threshold: float = 0.01
    max_iter: int = 10
    delta_std_ratio: float = 1.0
    min_abs_change: int = 0
    start_point: Optional[int] = None
    change_directions: Optional[List[str]] = None
    interest_window: Optional[int] = None
    magnitude_quantile: Optional[float] = None
    magnitude_ratio: float = 1.3
    magnitude_comparable_day: float = 0.5
    return_all_changepoints: bool = False
    remove_seasonality: bool = False


@dataclass
class CUSUMChangePointVal:
    changepoint: int
    mu0: float
    mu1: float
    changetime: float
    stable_changepoint: bool
    delta: float
    llr_int: float
    p_value_int: float
    delta_int: Optional[float]
    sigma0: Optional[float] = None
    sigma1: Optional[float] = None
    llr: Optional[float] = None
    p_value: Optional[float] = None
    regression_detected: Optional[bool] = None


class CUSUMChangePoint(TimeSeriesChangePoint):
    """CUSUM change point.

    This is a changepoint detected by CUSUMDetector.

    Attributes:

        start_time: Start time of the change.
        end_time: End time of the change.
        confidence: The confidence of the change point.
        direction: a str stand for the changepoint change direction 'increase'
            or 'decrease'.
        cp_index: an int for changepoint index.
        mu0: a float indicates the mean before changepoint.
        mu1: a float indicates the mean after changepoint.
        delta: mu1 - mu0.
        llr: log likelihood ratio.
        llr_int: log likelihood ratio in the interest window.
        regression_detected: a bool indicates if regression detected.
        stable_changepoint: a bool indicates if we have a stable changepoint
            when locating the changepoint.
        p_value: p_value of the changepoint.
        p_value_int: p_value of the changepoint in the interest window.
    """

    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        confidence: float,
        direction: str,
        cp_index: int,
        mu0: Union[float, np.ndarray],
        mu1: Union[float, np.ndarray],
        delta: Union[float, np.ndarray],
        llr_int: float,
        llr: float,
        regression_detected: bool,
        stable_changepoint: bool,
        p_value: float,
        p_value_int: float,
    ) -> None:
        super().__init__(start_time, end_time, confidence)
        self._direction = direction
        self._cp_index = cp_index
        self._mu0 = mu0
        self._mu1 = mu1
        self._delta = delta
        self._llr_int = llr_int
        self._llr = llr
        self._regression_detected = regression_detected
        self._stable_changepoint = stable_changepoint
        self._p_value = p_value
        self._p_value_int = p_value_int

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def cp_index(self) -> int:
        return self._cp_index

    @property
    def mu0(self) -> Union[float, np.ndarray]:
        return self._mu0

    @property
    def mu1(self) -> Union[float, np.ndarray]:
        return self._mu1

    @property
    def delta(self) -> Union[float, np.ndarray]:
        return self._delta

    @property
    def llr(self) -> float:
        return self._llr

    @property
    def llr_int(self) -> float:
        return self._llr_int

    @property
    def regression_detected(self) -> bool:
        return self._regression_detected

    @property
    def stable_changepoint(self) -> bool:
        return self._stable_changepoint

    @property
    def p_value(self) -> float:
        return self._p_value

    @property
    def p_value_int(self) -> float:
        return self._p_value_int

    def __repr__(self) -> str:
        return (
            f"CUSUMChangePoint(start_time: {self._start_time}, end_time: "
            f"{self._end_time}, confidence: {self._confidence}, direction: "
            f"{self._direction}, index: {self._cp_index}, delta: {self._delta}, "
            f"regression_detected: {self._regression_detected}, "
            f"stable_changepoint: {self._stable_changepoint}, mu0: {self._mu0}, "
            f"mu1: {self._mu1}, llr: {self._llr}, llr_int: {self._llr_int}, "
            f"p_value: {self._p_value}, p_value_int: {self._p_value_int})"
        )

    def __eq__(self, other: TimeSeriesChangePoint) -> bool:
        if not isinstance(other, CUSUMChangePoint):
            # don't attempt to compare against unrelated types
            raise NotImplementedError

        return (
            self._start_time == other._start_time
            and self._end_time == other._end_time
            and self._confidence == other._confidence
            and self._direction == other._direction
            and self._cp_index == other._cp_index
            and self._delta == other._delta
            and self._regression_detected == other._regression_detected
            and self._stable_changepoint == other._stable_changepoint
            and self._mu0 == other._mu0
            and self._mu1 == other._mu1
            and self._llr == other._llr
            and self._llr_int == other._llr_int
            and self._p_value == other._p_value
            # and self._p_value_int == other._p_value_int
        )

    def _almost_equal(self, x: float, y: float, round_int: int = 10) -> bool:
        return (
            x == y
            or round(x, round_int) == round(y, round_int)
            or round(abs((y - x) / x), round_int) == 0
        )

    def almost_equal(self, other: TimeSeriesChangePoint, round_int: int = 10) -> bool:
        """
        Compare if two CUSUMChangePoint objects are almost equal to each other.
        """

        if not isinstance(other, CUSUMChangePoint):
            # don't attempt to compare against unrelated types
            raise NotImplementedError

        res = [
            self._start_time == other._start_time,
            self._end_time == other._end_time,
            self._almost_equal(self._confidence, other._confidence, round_int),
            self._direction == other._direction,
            self._cp_index == other._cp_index,
            # pyre-ignore
            self._almost_equal(self._delta, other._delta, round_int),
            self._regression_detected == other._regression_detected,
            self._stable_changepoint == other._stable_changepoint,
            # pyre-ignore
            self._almost_equal(self._mu0, other._mu0, round_int),
            # pyre-ignore
            self._almost_equal(self._mu1, other._mu1, round_int),
            self._almost_equal(self._llr, other._llr, round_int),
            self._almost_equal(self._llr_int, other._llr_int, round_int),
            self._almost_equal(self._p_value, other._p_value, round_int),
        ]

        return all(res)


class CUSUMDetector():
    interest_window: Optional[Tuple[int, int]] = None
    magnitude_quantile: Optional[float] = None
    magnitude_ratio: Optional[float] = None
    changes_meta: Optional[Dict[str, Dict[str, Any]]] = None

    def __init__(
        self,
        data
    ) -> None:
        """
        Args:
            data: pandas dataframe; The input time series data.
        """
        self.data = data

    def _get_change_point(
        self, ts: np.ndarray, max_iter: int, start_point: int, change_direction: str
    ) -> CUSUMChangePointVal:
        """
        Find change point in the timeseries.
        """
        interest_window = self.interest_window

        # locate the change point using cusum method
        if change_direction == "increase":
            changepoint_func = np.argmin
            _log.debug("Detecting increase changepoint.")
        else:
            assert change_direction == "decrease"
            changepoint_func = np.argmax
            _log.debug("Detecting decrease changepoint.")
        n = 0
        # use the middle point as initial change point to estimate mu0 and mu1
        if interest_window is not None:
            ts_int = ts[interest_window[0] : interest_window[1]]
        else:
            ts_int = ts

        if start_point is None:
            cusum_ts = np.cumsum(ts_int - np.mean(ts_int))
            changepoint = min(changepoint_func(cusum_ts), len(ts_int) - 2)
        else:
            changepoint = start_point

        mu0 = mu1 = None
        # iterate until the changepoint converage
        while n < max_iter:
            n += 1
            mu0 = np.mean(ts_int[: (changepoint + 1)])
            mu1 = np.mean(ts_int[(changepoint + 1) :])
            mean = (mu0 + mu1) / 2
            # here is where cusum is happening
            cusum_ts = np.cumsum(ts_int - mean)
            next_changepoint = max(1, min(changepoint_func(cusum_ts), len(ts_int) - 2))
            if next_changepoint == changepoint:
                break
            changepoint = next_changepoint

        if n == max_iter:
            _log.info("Max iteration reached and no stable changepoint found.")
            stable_changepoint = False
        else:
            stable_changepoint = True

        # llr in interest window
        if interest_window is None:
            llr_int = np.inf
            pval_int = np.NaN
            delta_int = None
        else:
            # need to re-calculating mu0 and mu1 after the while loop
            mu0 = np.mean(ts_int[: (changepoint + 1)])
            mu1 = np.mean(ts_int[(changepoint + 1) :])

            llr_int = self._get_llr(ts_int, mu0, mu1, changepoint)
            pval_int = 1 - chi2.cdf(llr_int, 2)
            delta_int = mu1 - mu0
            changepoint += interest_window[0]

        # full time changepoint and mean
        # Note: here we are using whole TS
        mu0 = np.mean(ts[: (changepoint + 1)])
        mu1 = np.mean(ts[(changepoint + 1) :])


        return CUSUMChangePointVal(
            changepoint=changepoint,
            mu0=mu0,
            mu1=mu1,
            changetime=list(self.data["time"])[changepoint],
            stable_changepoint=stable_changepoint,
            delta=mu1 - mu0,
            llr_int=llr_int,
            p_value_int=pval_int,
            delta_int=delta_int,
        )

    def _get_llr(
        self,
        ts: np.ndarray,
        mu0: float,
        mu1: float,
        changepoint: int,
    ) -> float:
        """
        Calculate the log likelihood ratio
        """
        scale = np.sqrt(
            (
                np.sum((ts[: (changepoint + 1)] - mu0) ** 2)
                + np.sum((ts[(changepoint + 1) :] - mu1) ** 2)
            )
            / (len(ts) - 2)
        )
        mu_tilde, sigma_tilde = np.mean(ts), np.std(ts)

        if scale == 0:
            scale = sigma_tilde * 0.01

        llr = -2 * (
            self._log_llr(ts[: (changepoint + 1)], mu_tilde, sigma_tilde, mu0, scale)
            + self._log_llr(ts[(changepoint + 1) :], mu_tilde, sigma_tilde, mu1, scale)
        )
        return llr

    def _log_llr(
        self, x: np.ndarray, mu0: float, sigma0: float, mu1: float, sigma1: float
    ) -> float:
        """Helper function to calculate log likelihood ratio.

        This function calculate the log likelihood ratio of two Gaussian
        distribution log(l(0)/l(1)).

        Args:
            x: the data value.
            mu0: mean of model 0.
            sigma0: std of model 0.
            mu1: mean of model 1.
            sigma1: std of model 1.

        Returns:
            the value of log likelihood ratio.
        """

        return np.sum(
            np.log(sigma1 / sigma0)
            + 0.5 * (((x - mu1) / sigma1) ** 2 - ((x - mu0) / sigma0) ** 2)
        )

    def _magnitude_compare(self, ts: np.ndarray) -> float:
        """
        Compare daily magnitude to avoid daily seasonality false positives.
        """
        time = list(self.data["time"])
        interest_window = self.interest_window
        magnitude_ratio = self.magnitude_ratio
        if interest_window is None:
            raise ValueError("detect must be called first")
        assert magnitude_ratio is not None

        # get number of days in historical window
        days = (max(time) - min(time)).days

        # get interest window magnitude
        mag_int = self._get_time_series_magnitude(
            ts[interest_window[0] : interest_window[1]]
        )

        comparable_mag = 0

        for i in range(days):
            start_time = time[interest_window[0]] - pd.Timedelta(f"{i}D")
            end_time = time[interest_window[1]] - pd.Timedelta(f"{i}D")
            start_idx = time[time == start_time].index[0]
            end_idx = time[time == end_time].index[0]

            hist_int = self._get_time_series_magnitude(ts[start_idx:end_idx])
            if mag_int / hist_int >= magnitude_ratio:
                comparable_mag += 1

        return comparable_mag / days

    def _get_time_series_magnitude(self, ts: np.ndarray) -> float:
        """
        Calculate the magnitude of a time series.
        """
        magnitude = np.quantile(ts, self.magnitude_quantile, interpolation="nearest")
        return magnitude

    def detector(self, **kwargs: Any) -> Sequence[CUSUMChangePoint]:
        """
        Find the change point and calculate related statistics.

        Args:

            threshold: Optional; float; significance level, default: 0.01.
            max_iter: Optional; int, maximum iteration in finding the
                changepoint.
            delta_std_ratio: Optional; float; the mean delta have to larger than
                this parameter times std of the data to be consider as a change.
            min_abs_change: Optional; int; minimal absolute delta between mu0
                and mu1.
            start_point: Optional; int; the start idx of the changepoint, if
                None means the middle of the time series.
            change_directions: Optional; list<str>; a list contain either or
                both 'increase' and 'decrease' to specify what type of change
                want to detect.
            interest_window: Optional; list<int, int>, a list containing the
                start and end of interest windows where we will look for change
                points. Note that llr will still be calculated using all data
                points.
            magnitude_quantile: Optional; float; the quantile for magnitude
                comparison, if none, will skip the magnitude comparison.
            magnitude_ratio: Optional; float; comparable ratio.
            magnitude_comparable_day: Optional; float; maximal percentage of
                days can have comparable magnitude to be considered as
                regression.
            return_all_changepoints: Optional; bool; return all the changepoints
                found, even the insignificant ones.

        Returns:
            A list of CUSUMChangePoint.
        """
        defaultArgs = CUSUMDefaultArgs()
        # Extract all arg values or assign defaults from default vals constant
        threshold = kwargs.get("threshold", defaultArgs.threshold)
        max_iter = kwargs.get("max_iter", defaultArgs.max_iter)
        delta_std_ratio = kwargs.get("delta_std_ratio", defaultArgs.delta_std_ratio)
        min_abs_change = kwargs.get("min_abs_change", defaultArgs.min_abs_change)
        start_point = kwargs.get("start_point", defaultArgs.start_point)
        change_directions = kwargs.get(
            "change_directions", defaultArgs.change_directions
        )
        interest_window = kwargs.get("interest_window", defaultArgs.interest_window)
        magnitude_quantile = kwargs.get(
            "magnitude_quantile", defaultArgs.magnitude_quantile
        )
        magnitude_ratio = kwargs.get("magnitude_ratio", defaultArgs.magnitude_ratio)
        magnitude_comparable_day = kwargs.get(
            "magnitude_comparable_day", defaultArgs.magnitude_comparable_day
        )
        return_all_changepoints = kwargs.get(
            "return_all_changepoints", defaultArgs.return_all_changepoints
        )

        self.interest_window = interest_window
        self.magnitude_quantile = magnitude_quantile
        self.magnitude_ratio = magnitude_ratio

        # Use array to store the data
        ts = np.asarray(list(self.data["increase"]))
        ts = ts.astype("float64")
        changes_meta = {}

        if change_directions is None:
            change_directions = ["increase", "decrease"]

        for change_direction in change_directions:
            if change_direction not in {"increase", "decrease"}:
                raise ValueError(
                    "Change direction must be 'increase' or 'decrease.' "
                    f"Got {change_direction}"
                )

            change_meta = self._get_change_point(
                ts,
                max_iter=max_iter,
                start_point=start_point,
                change_direction=change_direction,
            )
            change_meta.llr = llr = self._get_llr(
                ts,
                change_meta.mu0,
                change_meta.mu1,
                change_meta.changepoint,
            )
            change_meta.p_value = 1 - chi2.cdf(llr, 2)

            # compare magnitude on interest_window and historical_window
            if np.min(ts) >= 0:
                if magnitude_quantile and interest_window:
                    change_ts = ts if change_direction == "increase" else -ts
                    mag_change = (
                        self._magnitude_compare(change_ts) >= magnitude_comparable_day
                    )
                else:
                    mag_change = True
            else:
                mag_change = True
                if magnitude_quantile:
                    _log.warning(
                        (
                            "The minimal value is less than 0. Cannot perform "
                            "magnitude comparison."
                        )
                    )

            if_significant = llr > chi2.ppf(1 - threshold, 2)
            if_significant_int = change_meta.llr_int > chi2.ppf(1 - threshold, 2)
            if change_direction == "increase":
                larger_than_min_abs_change = (
                    change_meta.mu0 + min_abs_change < change_meta.mu1
                )
            else:
                larger_than_min_abs_change = (
                    change_meta.mu0 > change_meta.mu1 + min_abs_change
                )
            larger_than_std = (
                np.abs(change_meta.delta)
                > np.std(ts[: change_meta.changepoint]) * delta_std_ratio
            )

            change_meta.regression_detected = (
                if_significant
                and if_significant_int
                and larger_than_min_abs_change
                and larger_than_std
                and mag_change
            )
            changes_meta[change_direction] = asdict(change_meta)

        self.changes_meta = changes_meta

        return self._convert_cusum_changepoints(changes_meta, return_all_changepoints)

    def _convert_cusum_changepoints(
        self,
        cusum_changepoints: Dict[str, Dict[str, Any]],
        return_all_changepoints: bool,
    ) -> List[CUSUMChangePoint]:
        """
        Convert the output from the other kats cusum algorithm into
        CUSUMChangePoint type.
        """
        converted = []
        detected_cps = cusum_changepoints

        for direction in detected_cps:
            dir_cps = detected_cps[direction]
            if dir_cps["regression_detected"] or return_all_changepoints:
                # we have a change point
                change_point = CUSUMChangePoint(
                    start_time=dir_cps["changetime"],
                    end_time=dir_cps["changetime"],
                    confidence=1 - dir_cps["p_value"],
                    direction=direction,
                    cp_index=dir_cps["changepoint"],
                    mu0=dir_cps["mu0"],
                    mu1=dir_cps["mu1"],
                    delta=dir_cps["delta"],
                    llr_int=dir_cps["llr_int"],
                    llr=dir_cps["llr"],
                    regression_detected=dir_cps["regression_detected"],
                    stable_changepoint=dir_cps["stable_changepoint"],
                    p_value=dir_cps["p_value"],
                    p_value_int=dir_cps["p_value_int"],
                )
                converted.append(change_point)

        return converted

