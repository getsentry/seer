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
from seer.anomaly_detection.detectors.smoothers import FlagSmoother
from seer.anomaly_detection.detectors.window_size_selectors import WindowSizeSelector
from seer.anomaly_detection.models import (
    AnomalyDetectionConfig,
    AnomalyFlags,
    MPTimeSeriesAnomalies,
    TimeSeries,
    TimeSeriesAnomalies,
)
from seer.dependency_injection import inject, injected
from seer.exceptions import ServerError

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
        flag_smoother: FlagSmoother = injected,
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
        if len(timeseries.values) == 0:
            raise ServerError("No values to detect anomalies for")
        if len(timeseries.timestamps) != len(timeseries.values):
            raise ServerError("Timestamps and values are not of the same length")

        ts_values = timeseries.values
        window_size = ws_selector.optimal_window_size(ts_values)
        if window_size <= 0:
            # TODO: Add sentry logging of this error
            raise ServerError("Invalid window size")
        # Get the matrix profile for the time series
        mp = stumpy.stump(
            ts_values,
            m=max(3, window_size),
            ignore_trivial=mp_config.ignore_trivial,
            normalize=False,
        )

        # We do not normalize the matrix profile here as normalizing during stream detection later is not straighforward.
        mp_dist = mp_utils.get_mp_dist_from_mp(mp, pad_to_len=len(ts_values))

        flags_and_scores = scorer.batch_score(
            values=ts_values,
            timestamps=timeseries.timestamps,
            mp_dist=mp_dist,
            ad_config=config,
            window_size=window_size,
        )
        if flags_and_scores is None:
            raise ServerError("Failed to score the matrix profile distance")

        # Apply smoothing to the flags
        smoothed_flags = flag_smoother.smooth(
            flags=flags_and_scores.flags,
            ad_config=config,
        )

        # Update the flags in flags_and_scores with the smoothed flags
        flags_and_scores.flags = smoothed_flags

        return MPTimeSeriesAnomalies(
            flags=flags_and_scores.flags,
            scores=flags_and_scores.scores,
            matrix_profile=mp,
            window_size=window_size,
            thresholds=flags_and_scores.thresholds,
        )


class MPStreamAnomalyDetector(AnomalyDetector):
    history_timestamps: npt.NDArray[np.float64] = Field(
        ..., description="Baseline timeseries to which streaming points will be added."
    )
    history_values: npt.NDArray[np.float64] = Field(
        ..., description="Baseline timeseries to which streaming points will be added."
    )
    history_mp: npt.NDArray[np.float64] = Field(
        ..., description="Matrix profile of the baseline timeseries."
    )
    window_size: int = Field(..., description="Window size to use for stream computation")
    history_flags: list[AnomalyFlags] = Field(..., description="Flags of the baseline timeseries.")
    anomaly_algo_data: list[dict] = Field(
        ..., description="Anomaly algo data of the baseline timeseries."
    )
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
        flag_smoother: FlagSmoother = injected,
    ) -> MPTimeSeriesAnomalies:
        """
        This method uses stumpy.stumpi to stream compute the matrix profile and scores the matrix profile distances

        parameters:
        timeseries: list[TimeSeriesPoint]
            The timeseries
        config: AnomalyDetectionConfig
            Parameters for tweaking the algorithm

        Returns:
            A MPTimeSeriesAnomalies object with the matrix profile, matrix profile distances, anomaly scores, anomaly flags and the window size used
        """
        if len(timeseries.values) == 0:
            raise ServerError("No values to detect anomalies for")
        if len(self.history_values) != len(self.history_timestamps):
            raise ServerError("History values and timestamps are not of the same length")

        if len(self.history_values) - self.window_size + 1 != len(self.history_mp):
            raise ServerError("Matrix profile is not of the right length.")
        stream = None
        with sentry_sdk.start_span(description="Initializing MP stream"):
            # Initialize stumpi
            stream = stumpy.stumpi(
                self.history_values,
                m=self.window_size,
                mp=self.history_mp,
                normalize=False,
                egress=False,
            )

        with sentry_sdk.start_span(description="Stream compute MP"):
            scores: list[float] = []
            flags: list[AnomalyFlags] = []
            streamed_mp: list[list[float]] = []
            thresholds: list[float] = []
            for cur_val, cur_timestamp in zip(timeseries.values, timeseries.timestamps):

                # Update the stumpi stream processor with new data
                stream.update(cur_val)

                # Get the matrix profile for the new data and score it
                cur_mp = [stream.P_[-1], stream.I_[-1], stream.left_I_[-1], -1]
                streamed_mp.append(cur_mp)
                mp_dist_baseline = mp_utils.get_mp_dist_from_mp(self.history_mp, pad_to_len=None)
                flags_and_scores = scorer.stream_score(
                    streamed_value=cur_val,
                    streamed_timestamp=cur_timestamp,
                    streamed_mp_dist=stream.P_[-1],
                    history_values=self.history_values,
                    history_timestamps=self.history_timestamps,
                    history_mp_dist=mp_dist_baseline,
                    ad_config=config,
                    window_size=self.window_size,
                )
                if flags_and_scores is None:
                    raise ServerError("Failed to score the matrix profile distance")

                # The original flags are the flags of the previous points
                past_original_flags = [
                    algo_data["original_flag"] for algo_data in self.anomaly_algo_data
                ]

                self.anomaly_algo_data.append({"original_flag": flags_and_scores.flags[-1]})

                # Apply stream smoothing to the newest flag based on the previous original flags
                smoothed_flags = flag_smoother.smooth(
                    flags=past_original_flags,
                    ad_config=config,
                    vote_threshold=0.3,
                    stream_smoothing=True,
                    cur_flag=flags_and_scores.flags,
                )

                original_flags = flags_and_scores.flags
                flags_and_scores.flags = smoothed_flags

                scores.extend(flags_and_scores.scores)
                flags.extend(flags_and_scores.flags)
                thresholds.extend(flags_and_scores.thresholds)

                # Add new data point as well as its matrix profile to baseline
                self.history_timestamps = np.append(self.history_timestamps, cur_timestamp)
                self.history_values = stream.T_
                self.history_mp = np.vstack([self.history_mp, cur_mp])

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
                thresholds=thresholds,
                original_flags=original_flags,
            )
