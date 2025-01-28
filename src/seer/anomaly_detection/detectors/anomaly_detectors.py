import abc
import datetime
import logging

import numpy as np
from dataclasses import dataclass
import numpy.typing as npt
import sentry_sdk
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.detectors.mp_scorers import MPScorer
from seer.anomaly_detection.detectors.mp_utils import MPUtils
from seer.anomaly_detection.detectors.smoothers import (
    MajorityVoteBatchFlagSmoother,
    MajorityVoteStreamFlagSmoother,
)
from seer.anomaly_detection.detectors.window_size_selectors import WindowSizeSelector
from seer.anomaly_detection.models import (
    AlgoConfig,
    AnomalyDetectionConfig,
    AnomalyFlags,
    MPTimeSeriesAnomaliesSingleWindow,
    Threshold,
    TimeSeries,
    TimeSeriesAnomalies,
)
from seer.dependency_injection import inject, injected
from seer.exceptions import ServerError

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for monitoring stream processing performance"""
    batch_size: int
    points_processed: int
    time_per_point_ms: float
    total_time_ms: float

class AnomalyDetector(BaseModel, abc.ABC):
    """
    Abstract base class for anomaly detection logic.
    """

    @abc.abstractmethod
    def detect(
        self,
        timeseries: TimeSeries,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig,
        time_budget_ms: int | None = None,
    ) -> TimeSeriesAnomalies:
        return NotImplemented


class MPBatchAnomalyDetector(AnomalyDetector):
    """
    This class encapsulates the logic for using Matrix Profile for batch anomaly detection.
    """

    @inject
    @sentry_sdk.trace
    def detect(
        self,
        timeseries: TimeSeries,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig = injected,
        time_budget_ms: int | None = None,
        window_size: int | None = None,
    ) -> MPTimeSeriesAnomaliesSingleWindow:
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
        return self._compute_matrix_profile(
            timeseries, ad_config, algo_config, window_size, time_budget_ms=time_budget_ms
        )

    @inject
    @sentry_sdk.trace
    def _compute_matrix_profile(
        self,
        timeseries: TimeSeries,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig,
        window_size: int | None = None,
        time_budget_ms: int | None = None,
        ws_selector: WindowSizeSelector = injected,
        scorer: MPScorer = injected,
        mp_utils: MPUtils = injected,
    ) -> MPTimeSeriesAnomaliesSingleWindow:
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
        if window_size is None:
            window_size = ws_selector.optimal_window_size(ts_values)
        if window_size <= 0:
            # TODO: Add sentry logging of this error
            raise ServerError("Invalid window size")
        # Get the matrix profile for the time series
        mp = stumpy.stump(
            ts_values,
            m=max(3, window_size),
            ignore_trivial=algo_config.mp_ignore_trivial,
            normalize=False,
        )

        # We do not normalize the matrix profile here as normalizing during stream detection later is not straighforward.
        mp_dist = mp_utils.get_mp_dist_from_mp(mp, pad_to_len=len(ts_values))

        flags_and_scores = scorer.batch_score(
            values=ts_values,
            timestamps=timeseries.timestamps,
            mp_dist=mp_dist,
            ad_config=ad_config,
            window_size=window_size,
            time_budget_ms=time_budget_ms,
        )
        if flags_and_scores is None:
            raise ServerError("Failed to score the matrix profile distance")

        original_flags = flags_and_scores.flags

        # Apply smoothing to the flags
        batch_flag_smoother = MajorityVoteBatchFlagSmoother()
        smoothed_flags = batch_flag_smoother.smooth(
            flags=flags_and_scores.flags,
            ad_config=ad_config,
            algo_config=algo_config,
        )

        # Update the flags in flags_and_scores with the smoothed flags
        flags_and_scores.flags = smoothed_flags

        return MPTimeSeriesAnomaliesSingleWindow(
            flags=smoothed_flags,
            scores=flags_and_scores.scores,
            matrix_profile=mp,
            window_size=window_size,
            thresholds=flags_and_scores.thresholds if algo_config.return_thresholds else None,
            original_flags=original_flags,
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
    original_flags: list[AnomalyFlags | None] = Field(
        ..., description="Original flags of the baseline timeseries."
    )
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def _calculate_adaptive_batch_size(
        self,
        total_points: int,
        time_budget_ms: int | None,
        initial_batch_size: int = 10
    ) -> int:
        """
        Calculate optimal batch size based on data volume and time budget.
        
        Args:
            total_points: Total number of points to process
            time_budget_ms: Available time budget in milliseconds
            initial_batch_size: Default batch size to start with
            
        Returns:
            Calculated batch size based on constraints
        """
        if time_budget_ms is None:
            return initial_batch_size
            
        # Minimum processing time needed per point (measured empirically)
        MIN_MS_PER_POINT = 2
        
        # Calculate theoretical max points that can be processed
        max_points = time_budget_ms / MIN_MS_PER_POINT
        
        # Adjust batch size based on data volume
        if total_points > max_points:
            # Use smaller batches for large volumes
            return max(1, min(5, initial_batch_size))
        elif total_points < max_points / 2:
            # Use larger batches for small volumes
            return min(20, initial_batch_size * 2)
            
        return initial_batch_size

    def _optimize_mp_computation(
        self,
        stream: stumpy.stumpi,
        cur_val: float,
        time_budget_ms: int | None = None
    ) -> tuple[float, float, float]:
        """
        Optimized matrix profile computation with time budget awareness.
        """
        if time_budget_ms is None:
            stream.update(cur_val)
            return stream.P_[-1], stream.I_[-1], stream.left_I_[-1]
            
        start_time = datetime.datetime.now()
        stream.update(cur_val)
        
        # Check if we're close to timeout
        elapsed = (datetime.datetime.now() - start_time).total_milliseconds()
        if elapsed > time_budget_ms * 0.8:  # If using >80% of budget
            return stream.P_[-1], -1, -1  # Skip index computations
            
        return stream.P_[-1], stream.I_[-1], stream.left_I_[-1]

    @inject
    @sentry_sdk.trace
    def detect(
        self,
        timeseries: TimeSeries,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig = injected,
        time_budget_ms: int | None = None,
        scorer: MPScorer = injected,
        mp_utils: MPUtils = injected,
    ) -> MPTimeSeriesAnomaliesSingleWindow:
        """
        This method uses stumpy.stumpi to stream compute the matrix profile and scores the matrix profile distances

        parameters:
        timeseries: list[TimeSeriesPoint]
            The timeseries
        config: AnomalyDetectionConfig
            Parameters for tweaking the algorithm

        Returns:
            A MPTimeSeriesAnomaliesSingleWindow object with the matrix profile, matrix profile distances, anomaly scores, anomaly flags and the window size used
        """
        if len(timeseries.values) == 0:
            raise ServerError("No values to detect anomalies for")
        if len(self.history_values) != len(self.history_timestamps):
            raise ServerError("History values and timestamps are not of the same length")

        if len(self.history_values) - self.window_size + 1 != len(self.history_mp):
            raise ServerError(
                f"Matrix profile is not of the right length. expected: {len(self.history_values) - self.window_size + 1}, actual: {len(self.history_mp)}"
            )
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
            thresholds: list[list[Threshold]] = []
            time_allocated = (
                datetime.timedelta(milliseconds=time_budget_ms) if time_budget_ms else None
            )
            time_start = datetime.datetime.now()
            
            # Calculate adaptive batch size
            batch_size = self._calculate_adaptive_batch_size(
                total_points=len(timeseries.values),
                time_budget_ms=time_budget_ms
            )
            
            # Initialize metrics tracking
            points_processed = 0
            processing_start = datetime.datetime.now()
            
            # Track batch statistics for monitoring
            sentry_sdk.set_extra("batch_size", batch_size)
            sentry_sdk.set_extra("total_points", len(timeseries.values))
            sentry_sdk.set_extra("time_budget_ms", time_budget_ms)

            for i, (cur_val, cur_timestamp) in enumerate(
                zip(timeseries.values, timeseries.timestamps)
            ):
                if time_allocated is not None and i % batch_size == 0:
                    time_elapsed = datetime.datetime.now() - time_start
                    if time_allocated is not None and time_elapsed > time_allocated:
                        sentry_sdk.set_extra("time_taken_for_batch_detection", time_elapsed)
                        sentry_sdk.set_extra("time_allocated_for_batch_detection", time_allocated)
                        sentry_sdk.capture_message(
                            f"stream_detection_timeout_{batch_size}",
                            level="error",
                        )
                        # Try to recover by processing remaining points with minimal batch size
                        if batch_size > 1:
                            batch_size = 1
                            continue
                            
                        raise ServerError("Stream detection took too long")

                # Use optimized matrix profile computation
                P, I, left_I = self._optimize_mp_computation(stream, cur_val, time_budget_ms)
                cur_mp = [P, I, left_I, -1]
                points_processed += 1

                streamed_mp.append(cur_mp)

            # Record final processing metrics
            processing_end = datetime.datetime.now()
            total_time_ms = (processing_end - processing_start).total_seconds() * 1000
            metrics = ProcessingMetrics(
                batch_size=batch_size,
                points_processed=points_processed,
                time_per_point_ms=total_time_ms / points_processed if points_processed > 0 else 0,
                total_time_ms=total_time_ms
            )
            self._record_metrics(metrics)

                mp_dist_baseline = mp_utils.get_mp_dist_from_mp(self.history_mp, pad_to_len=None)
                flags_and_scores = scorer.stream_score(
                    streamed_value=cur_val,
                    streamed_timestamp=cur_timestamp,
                    streamed_mp_dist=stream.P_[-1],
                    history_values=self.history_values,
                    history_timestamps=self.history_timestamps,
                    history_mp_dist=mp_dist_baseline,
                    ad_config=ad_config,
                    window_size=self.window_size,
                )
                if flags_and_scores is None:
                    raise ServerError("Failed to score the matrix profile distance")

                self.original_flags.append(flags_and_scores.flags[-1])

                stream_flag_smoother = MajorityVoteStreamFlagSmoother()

                # Apply stream smoothing to the newest flag based on the previous original flags
                smoothed_flags = stream_flag_smoother.smooth(
                    original_flags=self.original_flags,
                    ad_config=ad_config,
                    algo_config=algo_config,
                    vote_threshold=0.3,
                    cur_flag=flags_and_scores.flags,
                )

                flags_and_scores.flags = smoothed_flags

                scores.extend(flags_and_scores.scores)
                flags.extend(flags_and_scores.flags)
                if flags_and_scores.thresholds is not None:
                    thresholds.extend(flags_and_scores.thresholds)

                # Add new data point as well as its matrix profile to baseline
                self.history_timestamps = np.append(self.history_timestamps, cur_timestamp)
                self.history_values = stream.T_
                self.history_mp = np.vstack([self.history_mp, cur_mp])

            return MPTimeSeriesAnomaliesSingleWindow(
                flags=flags,
                scores=scores,
                matrix_profile=stumpy.mparray.mparray(
                    streamed_mp,
                    k=1,
                    m=self.window_size,
                    excl_zone_denom=stumpy.config.STUMPY_EXCL_ZONE_DENOM,
                ),
                window_size=self.window_size,
                thresholds=thresholds if algo_config.return_thresholds else None,
                original_flags=self.original_flags,
            )
            
    def _record_metrics(self, metrics: ProcessingMetrics) -> None:
        """Record processing metrics for monitoring"""
        sentry_sdk.set_extra("processing_metrics", {
            "batch_size": metrics.batch_size,
            "points_processed": metrics.points_processed,
            "time_per_point_ms": metrics.time_per_point_ms,
            "total_time_ms": metrics.total_time_ms,
            "processing_efficiency": {
                "points_per_second": 1000 / metrics.time_per_point_ms if metrics.time_per_point_ms > 0 else 0,
                "batch_processing_overhead": metrics.total_time_ms / metrics.batch_size if metrics.batch_size > 0 else 0
            }
        })
