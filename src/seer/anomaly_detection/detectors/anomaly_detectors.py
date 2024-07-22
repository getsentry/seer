import abc
import logging

import numpy as np
import numpy.typing as npt
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel, Field

from seer.anomaly_detection.detectors.mp_scorers import MPScorer
from seer.anomaly_detection.detectors.normalizers import Normalizer
from seer.anomaly_detection.detectors.window_size_selectors import WindowSizeSelector
from seer.anomaly_detection.models.internal import MPTimeSeriesAnomalies, TimeSeries

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
    def detect(self, timeseries: TimeSeries) -> TimeSeries:
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

    def detect(self, timeseries: TimeSeries) -> TimeSeries:
        """
        This method uses matrix profile to detect and score anonalies in the time series.

        Parameters:
        timeseries: TimeSeries
            The timeseries

        Returns:
        The input timeseries with an anomaly scores and a flag added
        """
        anomalies = self._compute_matrix_profile(timeseries)
        timeseries.anomalies = anomalies

        return timeseries

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

    def _compute_matrix_profile(self, timeseries: TimeSeries) -> MPTimeSeriesAnomalies:
        """
        This method calls stumpy.stump to compute the matrix profile and scores the matrix profile distances

        parameters:
        timeseries: list[TimeSeriesPoint]
            The timeseries

        Returns:
            A tuple with the matrix profile, matrix profile distances, anomaly scores, anomaly flags and the window size used

        """
        ts_values = timeseries.values  # np.array([np.float64(point.value) for point in timeseries])
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

        scores, flags = self.scorer.score(mp, mp_dist, 0)

        return MPTimeSeriesAnomalies(
            flags=flags,
            scores=scores,
            matrix_profile=mp,
            window_size=window_size,
        )


# class MPStreamAnomalyDetector(AnomalyDetector):
#     config: MPConfig = Field(..., description="Configuration for the algorithm")
#     scorer: MPScorer = Field(
#         ..., description="The scorer to use for evaluating if a point is an anomaly or not"
#     )
#     normalizer: Normalizer = Field(..., description="Normalizer to use for normalizing data")
#     base_timeseries: list[TimeSeriesPoint] = Field(..., description="Baseline timeseries to which streaming points will be added.")
#     base_mp: npt.NDArray = Field(..., description="Matrix profile of the baseline timeseries.")
#     window_size: int = Field(..., description="Window size to use for stream computation")

#     def detect(self, timeseries: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
#         # Initialize stumpi
#         stream = stumpy.stumpi(ts, m=self.window_size, mp=self.base_mp, normalize=False, egress=False)
#         for point in timeseries:
#             # Evaluate MP for cur value
#             stream.update(point.value)
#             cur_mp = [stream.P_[-1], stream.I_[-1], stream.left_I_[-1], -1]
#             cur_mp_dist = stream.P_[-1]
#             self.base_timeseries.append(point)
#             self.base_mp = np.append(self.base_mp, cur_mp, axis=1)

#             scores, flags = self.scorer.score([point.value], [cur_mp], [cur_mp_dist], self.window_size)

#         # For each input point, add to stumpi and score it
#         return [
#             TimeSeriesPoint(
#                 timestamp=point.timestamp,
#                 value=point.value,
#                 anomaly=Anomaly(anomaly_type="none", anomaly_score=0.5),
#             )
#             for point in timeseries or []
#         ]


class DummyAnomalyDetector(AnomalyDetector):
    """
    Dummy anomaly detector used during dev work
    """

    def detect(self, timeseries: TimeSeries) -> TimeSeries:
        anomalies = MPTimeSeriesAnomalies(
            flags=np.array(["none"] * len(timeseries.values)),
            scores=np.array([np.float64(0.5)] * len(timeseries.values)),
            matrix_profile=np.array([]),
            window_size=0,
        )
        return TimeSeries(
            timestamps=timeseries.timestamps, values=timeseries.values, anomalies=anomalies
        )
