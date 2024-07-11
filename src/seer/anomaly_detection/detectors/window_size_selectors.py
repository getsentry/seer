import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field

from seer.anomaly_detection.detectors.normalizers import MinMaxNormalizer, Normalizer


class WindowSizeSelector(BaseModel):
    def optimal_window_size(self, time_series: npt.NDArray) -> int:
        raise NotImplementedError("Subclasses should implement this!")


class SuSSWindowSizeSelector(WindowSizeSelector):
    lbound: int = Field(10)
    threshold: float = Field(0.89)
    normalizer: Normalizer = Field(MinMaxNormalizer())

    def _score(self, time_series, window_size, stats) -> np.float64:
        roll = pd.Series(time_series).rolling(window_size)
        ts_mean, ts_std, ts_min_max = stats

        roll_mean = roll.mean().to_numpy()[window_size:]
        roll_std = roll.std(ddof=0).to_numpy()[window_size:]
        roll_min = roll.min().to_numpy()[window_size:]
        roll_max = roll.max().to_numpy()[window_size:]

        X = np.array([roll_mean - ts_mean, roll_std - ts_std, (roll_max - roll_min) - ts_min_max])

        X = np.sqrt(np.sum(np.square(X), axis=0)) / np.sqrt(window_size)
        score = np.mean(X)
        return score

    def optimal_window_size(self, time_series: npt.NDArray) -> int:
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

        # exponential search (to find window size interval)
        while True:
            window_size = 2**exp

            if window_size < lbound:
                exp += 1
                continue

            score = 1 - (self._score(time_series, window_size, stats) - min_score) / (
                max_score - min_score
            )

            if score > threshold:
                break

            exp += 1

        lbound, ubound = max(lbound, 2 ** (exp - 1)), 2**exp + 1

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
