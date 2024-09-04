import abc

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field

from seer.anomaly_detection.detectors.normalizers import MinMaxNormalizer, Normalizer


class WindowSizeSelector(BaseModel, abc.ABC):
    """
    Abstract base class for window size selection logic
    """

    @abc.abstractmethod
    def optimal_window_size(self, time_series: npt.NDArray[np.float64]) -> int:
        return NotImplemented


class SuSSWindowSizeSelector(WindowSizeSelector):
    """
    This class uses the Summary Statistics Subsequence (SuSS) algorithm to find optimal window size. SuSS compares
    summary statistics - mean, standard deviation and range -  computed over windows with the ones of the entire TS.
    Intuition is that summary statistics of appropriate window size are close to those of the whole signal. It performs
    exponential search followed by a binary search to locate window size with SuSS score larger then pre-defined
    threshold (fixed to 89%). This implementation as well as the default values for lower bound and threshold are
    based on the paper titled "Window Size Selection In Unsupervised Time Series Analytics: A Review and Benchmark".

    References:
    Arik Ermshaus, Patrick Schäfer, and Ulf Leser. 2023. Window Size Selection in Unsupervised Time Series Analytics: A
    Review and Benchmark. In Advanced Analytics and Learning on Temporal Data:
    7th ECML PKDD Workshop, AALTD 2022, Grenoble, France, September 19–23,
    2022, Revised Selected Papers. Springer-Verlag, Berlin, Heidelberg, 83–101.
    https://doi.org/10.1007/978-3-031-24378-3_6

    """

    lbound: int = Field(10, description="Determines the minimum window size to look for")
    threshold: float = Field(
        0.89, description="Score threshold to use to find the best window size"
    )
    normalizer: Normalizer = Field(
        MinMaxNormalizer(), description="Normalizer to use for normnalizing timeseries"
    )

    def _score(
        self, time_series: npt.NDArray[np.float64], window_size: int, stats: tuple
    ) -> np.float64:
        """
        Calculates the overall score for a given window size. It does this by
        * splitting the time series into subsequences of the given window length
        * computing summary statistics for each subsequence
        * calculating the distance between the local summary statistics and the global summary statistics
        * taking the mean of all scores

        Parameters:
        time_series: np.float64
            The time series as a seaquence of float values.

        """
        roll = pd.Series(time_series).rolling(window_size)
        ts_mean, ts_std, ts_min_max = stats

        roll_mean = roll.mean().to_numpy()[window_size:]
        roll_std = roll.std(ddof=0).to_numpy()[window_size:]
        roll_min = roll.min().to_numpy()[window_size:]
        roll_max = roll.max().to_numpy()[window_size:]

        scores = np.array(
            [roll_mean - ts_mean, roll_std - ts_std, (roll_max - roll_min) - ts_min_max]
        )
        scores = np.sqrt(np.sum(np.square(scores), axis=0)) / np.sqrt(window_size)
        mean_score = np.mean(scores)
        return mean_score

    def optimal_window_size(self, time_series: npt.NDArray[np.float64]) -> int:
        """
        This method applies the SuSS algorithm to find the optimal window size. SuSS algoright is covered in more detail
        in the class documentation.

        parameters:
        time_series: np.float64
            The time series as a seaquence of float values.

        """
        time_series = self.normalizer.normalize(time_series)
        ts_mean = np.mean(time_series)
        ts_std = np.std(time_series)
        ts_min_max = np.max(time_series) - np.min(time_series)

        lbound = self.lbound
        threshold = self.threshold

        stats = (ts_mean, ts_std, ts_min_max)
        max_score = self._score(time_series, 1, stats)
        min_score = self._score(time_series, time_series.shape[0] - 1, stats)

        exp = 0
        found_window = False

        # exponential search (to find window size interval) with limit to avoid overflow
        while 2**exp < np.iinfo(np.integer).max:
            window_size = 2**exp

            if window_size < lbound:
                exp += 1
                continue

            score = 1 - (self._score(time_series, window_size, stats) - min_score) / (
                max_score - min_score
            )

            if score > threshold:
                found_window = True
                break

            exp += 1

        if not found_window:
            raise Exception("Search for optimal window failed.")

        lbound, ubound = max(lbound, 2 ** (exp - 1)), min(2**exp + 1, np.iinfo(np.integer).max)

        # binary search (to find window size in interval)
        while lbound <= ubound:
            window_size = int((lbound + ubound) / 2)
            score = 1 - (self._score(time_series, window_size, stats) - min_score) / (
                max_score - min_score
            )

            if score < threshold:
                lbound = window_size + 1
            elif score > threshold:
                ubound = window_size - 1
            else:
                break

        return 2 * lbound
