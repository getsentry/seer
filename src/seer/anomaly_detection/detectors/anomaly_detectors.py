import abc
import logging
import sentry_sdk

import numpy as np
import numpy.typing as npt
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.detectors.mp_config import MPConfig
from seer.anomaly_detection.detectors.mp_scorers import MPScorer
from seer.anomaly_detection.detectors.mp_utils import MPUtils
from seer.anomaly_detection.detectors.window_size_selectors import WindowSizeSelector
from seer.anomaly_detection.models import MPTimeSeriesAnomalies, TimeSeries, TimeSeriesAnomalies
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class AnomalyDetector(BaseModel, abc.ABC):
    """
    Abstract base class for anomaly detection logic.
    """

    @abc.abstractmethod
    def detect(self, timeseries: TimeSeries) -> TimeSeriesAnomalies:
        return NotImplemented


class MPBatchAnomalyDetector(AnomalyDetector):
    """
    This class encapsulates the logic for using Matrix Profile for batch anomaly detection.
    """

    @sentry_sdk.trace
    def detect(self, timeseries: TimeSeries) -> MPTimeSeriesAnomalies:
        """
        This method uses matrix profile to detect and score anonalies in the time series.

        Parameters:
        timeseries: TimeSeries
            The timeseries

        Returns:
        The input timeseries with an anomaly scores and a flag added
        """
        try:
            return self._compute_matrix_profile(timeseries)
        except Exception as e:
            sentry_sdk.capture_exception(e)

    @sentry_sdk.trace
    @inject
    def _compute_matrix_profile(
        self,
        timeseries: TimeSeries,
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

        scores, flags = scorer.score(mp_dist, mp_dist_baseline=None)

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
    base_mp: npt.NDArray = Field(..., description="Matrix profile of the baseline timeseries.")
    window_size: int = Field(..., description="Window size to use for stream computation")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @inject
    @sentry_sdk.trace
    def detect(
        self, timeseries: TimeSeries, scorer: MPScorer = injected, mp_utils: MPUtils = injected
    ) -> MPTimeSeriesAnomalies:
        # Initialize stumpi
        stream = stumpy.stumpi(
            self.base_values,
            m=self.window_size,
            mp=self.base_mp,
            normalize=False,
            egress=False,
        )

        scores = []
        flags = []
        for cur_val in timeseries.values:
            # Update the sumpi stream processor with new data
            stream.update(cur_val)

            # Get the matrix profile for the new data and score it
            cur_mp = [stream.P_[-1], stream.I_[-1], stream.left_I_[-1], -1]
            mp_dist_baseline = mp_utils.get_mp_dist_from_mp(self.base_mp, pad_to_len=None)
            cur_scores, cur_flags = scorer.score(
                mp_dist_to_score=np.array([stream.P_[-1]]), mp_dist_baseline=mp_dist_baseline
            )
            scores.extend(cur_scores)
            flags.extend(cur_flags)

            # Add new data point as well as its matrix profile to baseline
            self.base_values = stream.T_
            self.base_mp = np.vstack([self.base_mp, cur_mp])

        return MPTimeSeriesAnomalies(
            flags=flags,
            scores=scores,
            matrix_profile=self.base_mp,
            window_size=self.window_size,
        )


class DummyAnomalyDetector(AnomalyDetector):
    """
    Dummy anomaly detector used during dev work
    """

    def detect(self, timeseries: TimeSeries) -> TimeSeriesAnomalies:
        anomalies = MPTimeSeriesAnomalies(
            flags=np.array(["none"] * len(timeseries.values)),
            scores=np.array([np.float64(0.5)] * len(timeseries.values)),
            matrix_profile=np.array([]),
            window_size=0,
        )
        return anomalies
