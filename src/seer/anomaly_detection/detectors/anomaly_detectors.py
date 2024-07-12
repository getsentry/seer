import abc
import logging

import numpy as np
import numpy.typing as npt
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel, Field

from seer.anomaly_detection.detectors.mp_scorers import MPScorer
from seer.anomaly_detection.detectors.normalizers import Normalizer
from seer.anomaly_detection.detectors.window_size_selectors import WindowSizeSelector
from seer.anomaly_detection.models import Anomaly, TimeSeriesPoint

logger = logging.getLogger("anomaly_detection")


class MPConfig(BaseModel):
    """
    Class with configuration used for the Matrix Profile algorithm
    """

    ignore_trivial: bool = Field(
        ...,
        description="Flag that tells the stumpy library to ignore trivial matches to speed up MP computation",
    )
    normalize_mp: bool = Field(
        ...,
        description="Flag to control if the matrix profile is normalized first",
    )


class AnomalyDetector(BaseModel, abc.ABC):
    """
    Abstract base class for anomaly detection logic.
    """

    @abc.abstractmethod
    def detect(self, timeseries: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        return NotImplemented


class MPBatchAnomalyDetector(AnomalyDetector):
    """
    This class encapsulates the logic for using Matrix Profile for batch anomaly detection.
    """

    config: MPConfig = Field(..., description="Configuration for the algorithm")
    scorer: MPScorer = Field(
        ..., description="The scorer to use for evaluating if a point is an anomaly or not"
    )
    ws_selector: WindowSizeSelector = Field(
        ..., description="Window size selector logic to use for selecting the optimal window size"
    )
    normalizer: Normalizer = Field(..., description="Normalizer to use for normalizing data")

    def detect(self, timeseries: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        """
        This method uses matrix profile to detect and score anonalies in the time series.

        Parameters:
        timeseries: list[TimeSeriesPoint]
            The timeseries

        Returns:
        The timeseries with each point of the timeseries updated with an anomaly scores and a flag
        """
        mp, mp_dist, scores, flags, window_size = self._compute_matrix_profile(timeseries)

        self._update_ts(timeseries, scores, flags)
        return timeseries

    def _update_ts(
        self, timeseries: list[TimeSeriesPoint], scores: npt.NDArray, flags: list
    ) -> list[TimeSeriesPoint]:
        """
        Update the timeseries with score and flag. This method does an inplace update.
        """
        for point, score, flag in zip(timeseries, scores, flags):
            score = 0.0 if np.isnan(score) or score < 0 else score
            point.anomaly = Anomaly(anomaly_score=score, anomaly_type=flag)

    def _get_mp_dist_from_mp(
        self,
        mp: npt.NDArray,
        ts: npt.NDArray[np.float64],
        normalize_mp_dist: bool,
        pad_to_ts_len: bool,
    ) -> npt.NDArray[np.float64]:
        """
        Helper method for extracting the matrix profile distances from the matrix profile returned by the
        stumpy library

        Parameters:
        mp: Numpy array
            Contains the matrix returned by the stumpy.stump call

        ts: Numpy array
            The timeseries

        normalize_mp_dist: bool
            Flag indicating if the returned mp distances should be normalized. Normalization ensures that the
            distances are always between 0 and 1 (both values included)

        pad_to_ts_len: bool
            Flag that indicates of the distances should be padded with NaN to match the length of the time series

        Returns:
        The distances as a numpy array of floats
        """
        mp_dist = mp[:, 0]
        if normalize_mp_dist:
            mp_dist = self.normalizer.normalize(mp_dist)

        if pad_to_ts_len:
            nan_value_count = np.empty(len(ts) - len(mp_dist))
            nan_value_count.fill(np.nan)
            mp_dist_updated = np.concatenate((nan_value_count, mp_dist))
            return mp_dist_updated.astype(np.float64)
        else:
            return mp_dist.astype(float)

    def _compute_matrix_profile(self, timeseries: list[TimeSeriesPoint]) -> tuple:
        """
        This method calls stumpy.stump to compute the matrix profile and scores the matrix profile distances

        parameters:
        timeseries: list[TimeSeriesPoint]
            The timeseries

        Returns:
            A tuple with the matrix profile, matrix profile distances, anomaly scores, anomaly flags and the window size used

        """
        ts_values = np.array([np.float64(point.value) for point in timeseries])
        window_size = self.ws_selector.optimal_window_size(ts_values)
        logger.debug(f"window_size: {window_size}")
        if window_size <= 0:
            # TODO: Add sentry logging of this error
            raise Exception("Invalid window size")
        # Get the matrix profile for the time series
        mp = stumpy.stump(
            ts_values,
            m=max(3, window_size),
            ignore_trivial=self.config.ignore_trivial,
            normalize=False,
        )

        mp_dist = self._get_mp_dist_from_mp(
            mp, ts_values, self.config.normalize_mp, pad_to_ts_len=True
        )

        scores, flags = self.scorer.score(ts_values, mp, mp_dist, window_size)
        return mp, mp_dist, scores, flags, window_size


class DummyAnomalyDetector(AnomalyDetector):
    """
    Dummy anomaly detector used during dev work
    """

    def detect(self, timeseries: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        return [
            TimeSeriesPoint(
                timestamp=point.timestamp,
                value=point.value,
                anomaly=Anomaly(anomaly_type="none", anomaly_score=0.5),
            )
            for point in timeseries or []
        ]
