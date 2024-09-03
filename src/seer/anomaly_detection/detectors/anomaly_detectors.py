import abc
import logging

import numpy as np
import numpy.typing as npt
import sentry_sdk
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.detectors.mp_config import MPConfig
from seer.anomaly_detection.detectors.mp_scorers import MPScorer
from seer.anomaly_detection.detectors.mp_utils import MPUtils
from seer.anomaly_detection.detectors.window_size_selectors import WindowSizeSelector
from seer.anomaly_detection.models import (
    AnomalyDetectionConfig,
    MPTimeSeriesAnomalies,
    TimeSeries,
    TimeSeriesAnomalies,
)
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class AnomalyDetector(BaseModel, abc.ABC):
    """
    Abstract base class for anomaly detection logic.
    """

    @abc.abstractmethod
    def detect(self, timeseries: TimeSeries, config: AnomalyDetectionConfig) -> TimeSeriesAnomalies:
        return NotImplemented


class MPBatchAnomalyDetector(AnomalyDetector):
    """
    This class encapsulates the logic for using Matrix Profile for batch anomaly detection.
    """

    @sentry_sdk.trace
    def detect(
        self, timeseries: TimeSeries, config: AnomalyDetectionConfig
    ) -> MPTimeSeriesAnomalies:
        """
        This method uses matrix profile to detect and score anonalies in the time series.

        Parameters:
        timeseries: TimeSeries
            The timeseries

        config: AnomalyDetectionConfig
            Parameters for tweaking the algorithm

        Returns:
        The input timeseries with an anomaly scores and a flag added
        """
        return self._compute_matrix_profile(timeseries, config)

    @inject
    @sentry_sdk.trace
    def _compute_matrix_profile(
        self,
        timeseries: TimeSeries,
        config: AnomalyDetectionConfig,
        ws_selector: WindowSizeSelector = injected,
        mp_config: MPConfig = injected,
        scorer: MPScorer = injected,
        mp_utils: MPUtils = injected,
    ) -> MPTimeSeriesAnomalies:
        """
        This method calls stumpy.stump to compute the matrix profile and scores the matrix profile distances

        parameters:
        timeseries: list[TimeSeriesPoint]
            The timeseries
        config: AnomalyDetectionConfig
            Parameters for tweaking the algorithm

        Returns:
            A tuple with the matrix profile, matrix profile distances, anomaly scores, anomaly flags and the window size used

        """
        ts_values = timeseries.values  # np.array([np.float64(point.value) for point in timeseries])
        window_size = ws_selector.optimal_window_size(ts_values)
        logger.debug(f"window_size: {window_size}")
        if window_size <= 0:
            # TODO: Add sentry logging of this error
            raise Exception("Invalid window size")
        # Get the matrix profile for the time series
        mp = stumpy.stump(
            ts_values,
            m=max(3, window_size),
            ignore_trivial=mp_config.ignore_trivial,
            normalize=False,
        )

        # we do not normalize the matrix profile here as normalizing during stream detection later is not straighforward.
        mp_dist = mp_utils.get_mp_dist_from_mp(mp, pad_to_len=len(ts_values))

        scores, flags = scorer.batch_score(
            ts_values,
            mp_dist,
            sensitivity=config.sensitivity,
            direction=config.direction,
            window_size=window_size,
        )

        return MPTimeSeriesAnomalies(
            flags=flags,
            scores=scores,
            matrix_profile=mp,
            window_size=window_size,
        )


class MPStreamAnomalyDetector(AnomalyDetector):
    base_timestamps: npt.NDArray[np.float64] = Field(
        ..., description="Baseline timeseries to which streaming points will be added."
    )
    base_values: npt.NDArray[np.float64] = Field(
        ..., description="Baseline timeseries to which streaming points will be added."
    )
    base_mp: npt.NDArray[np.float64] = Field(
        ..., description="Matrix profile of the baseline timeseries."
    )
    window_size: int = Field(..., description="Window size to use for stream computation")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @inject
    @sentry_sdk.trace
    def detect(
        self,
        timeseries: TimeSeries,
        config: AnomalyDetectionConfig,
        scorer: MPScorer = injected,
        mp_utils: MPUtils = injected,
    ) -> MPTimeSeriesAnomalies:
        stream = None
        with sentry_sdk.start_span(description="Initializing MP stream"):
            # Initialize stumpi
            stream = stumpy.stumpi(
                self.base_values,
                m=self.window_size,
                mp=self.base_mp,
                normalize=False,
                egress=False,
            )

        with sentry_sdk.start_span(description="Stream compute MP"):
            scores = []
            flags = []
            streamed_mp = []
            for cur_val in timeseries.values:
                # Update the sumpi stream processor with new data
                stream.update(cur_val)

                # Get the matrix profile for the new data and score it
                cur_mp = [stream.P_[-1], stream.I_[-1], stream.left_I_[-1], -1]
                streamed_mp.append(cur_mp)
                mp_dist_baseline = mp_utils.get_mp_dist_from_mp(self.base_mp, pad_to_len=None)
                cur_scores, cur_flags = scorer.stream_score(
                    ts_value=cur_val,
                    mp_dist=stream.P_[-1],
                    sensitivity=config.sensitivity,
                    direction=config.direction,
                    window_size=self.window_size,
                    ts_baseline=self.base_values,
                    mp_dist_baseline=mp_dist_baseline,
                )
                scores.extend(cur_scores)
                flags.extend(cur_flags)

                # Add new data point as well as its matrix profile to baseline
                self.base_values = stream.T_
                self.base_mp = np.vstack([self.base_mp, cur_mp])

            return MPTimeSeriesAnomalies(
                flags=flags,
                scores=scores,
                matrix_profile=stumpy.mparray.mparray(
                    streamed_mp,
                    k=1,
                    m=self.window_size,
                    excl_zone_denom=stumpy.config.STUMPY_EXCL_ZONE_DENOM,
                ),
                window_size=self.window_size,
            )


class DummyAnomalyDetector(AnomalyDetector):
    """
    Dummy anomaly detector used during dev work
    """

    def detect(self, timeseries: TimeSeries, config: AnomalyDetectionConfig) -> TimeSeriesAnomalies:
        anomalies = MPTimeSeriesAnomalies(
            flags=np.array(["none"] * len(timeseries.values)),
            scores=np.array([np.float64(0.5)] * len(timeseries.values)),
            matrix_profile=np.array([]),
            window_size=0,
        )
        return anomalies
