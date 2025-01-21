import abc
import datetime
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import sentry_sdk
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.detectors.location_detectors import LocationDetector
from seer.anomaly_detection.models import (
    AlgoConfig,
    AnomalyDetectionConfig,
    AnomalyFlags,
    Directions,
    PointLocation,
    Sensitivities,
    Threshold,
    ThresholdType,
)
from seer.dependency_injection import inject, injected
from seer.exceptions import ClientError, ServerError
from seer.tags import AnomalyDetectionTags

logger = logging.getLogger(__name__)


class FlagsAndScores(BaseModel):
    flags: List[AnomalyFlags]
    scores: List[float]
    thresholds: List[List[Threshold]]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class MPScorer(BaseModel, abc.ABC):
    """
    Abstract class for scoring and flagging anomalies using matrix profile distances.
    """

    @abc.abstractmethod
    @inject
    def batch_score(
        self,
        values: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        time_budget_ms: int | None = None,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> Optional[FlagsAndScores]:
        return NotImplemented

    @abc.abstractmethod
    @inject
    def stream_score(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        streamed_mp_dist: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
        history_mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> Optional[FlagsAndScores]:
        return NotImplemented


class LowVarianceScorer(MPScorer):
    """
    This class implements a scoring method for detecting anomalies in time series data with low variance.
    It uses a simple threshold-based approach, comparing each value to the mean of the time series,
    scaled by a factor that depends on the desired sensitivity level. This method is particularly
    useful when the matrix profile approach might not be effective due to low variability in the data.
    """

    std_threshold: float = Field(
        0.01,
        description="Minimum standard deviation required in order to use IQR based scoring",
    )
    scaling_factors: Dict[Sensitivities, int] = Field(
        {
            # High sensitivity = more anomalies + higher false positives
            # 1.5x the mean and above is considered an anomaly
            "high": 1.5,
            # Medium sensitivity = lesser anomalies + lesser false positives
            # 3x the mean and above is considered an anomaly
            "medium": 3,
            # Low sensitivity = leaset anomalies + leaset false positives
            # 5x the mean and above is considered an anomaly
            "low": 5,
        },
        description="Lower and upper bounds for high sensitivity",
    )

    def _to_flag_and_score(
        self,
        val: np.float64,
        ts_mean: np.float64,
        ad_config: AnomalyDetectionConfig,
    ) -> tuple[AnomalyFlags, float, float, float]:
        if ad_config.sensitivity not in self.scaling_factors:
            raise ClientError(f"Invalid sensitivity: {ad_config.sensitivity}")
        scaling_factor = self.scaling_factors[ad_config.sensitivity]
        bound1 = ts_mean + ts_mean * scaling_factor
        bound2 = ts_mean - ts_mean * scaling_factor
        lower_bound: float = float(min(bound1, bound2))
        upper_bound: float = float(max(bound1, bound2))

        # if current value is significantly higher or lower than the mean then mark it as high anomaly else mark it as no anomaly
        if ad_config.direction == "both" and (val < lower_bound or val > upper_bound):
            return "anomaly_higher_confidence", 0.9, upper_bound, lower_bound
        elif ad_config.direction == "down" and val < lower_bound:
            return "anomaly_higher_confidence", 0.9, upper_bound, lower_bound
        elif ad_config.direction == "up" and val > upper_bound:
            return "anomaly_higher_confidence", 0.9, upper_bound, lower_bound
        return "none", 0.0, upper_bound, lower_bound

    @inject
    def batch_score(
        self,
        values: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        time_budget_ms: int | None = None,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> Optional[FlagsAndScores]:
        ts_mean = values.mean()
        scores = []
        flags = []
        thresholds = []
        if values.std() > self.std_threshold:
            sentry_sdk.set_tag(AnomalyDetectionTags.LOW_VARIATION_TS, 0)
            return None

        sentry_sdk.set_tag(AnomalyDetectionTags.LOW_VARIATION_TS, 1)
        for val in values:
            flag, score, pred_up, pred_down = self._to_flag_and_score(val, ts_mean, ad_config)
            flags.append(flag)
            scores.append(score)
            thresholds.append(
                [
                    Threshold(
                        type=ThresholdType.LOW_VARIANCE_THRESHOLD, upper=pred_up, lower=pred_down
                    )
                ]
            )
        return FlagsAndScores(flags=flags, scores=scores, thresholds=thresholds)

    @inject
    def stream_score(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        streamed_mp_dist: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
        history_mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> Optional[FlagsAndScores]:
        context = history_values[-2 * window_size :]
        if context.std() > self.std_threshold:
            return None
        flag, score, pred_up, pred_down = self._to_flag_and_score(
            streamed_value, context.mean(), ad_config
        )
        threshold = Threshold(
            type=ThresholdType.LOW_VARIANCE_THRESHOLD, upper=pred_up, lower=pred_up
        )

        return FlagsAndScores(
            flags=[flag],
            scores=[score],
            thresholds=[[threshold]],
        )


class MPIQRScorer(MPScorer):
    """
    This class implements a scoring method for detecting anomalies in time series data using the interquartile range (IQR) of the matrix profile distances.

    The IQR method is used to identify outliers in the matrix profile distances. It works by:
    1. Calculating the first quartile (Q1) and third quartile (Q3) of the matrix profile distances.
    2. Computing the IQR as Q3 - Q1.
    3. Defining a range of "normal" values based on the IQR and sensitivity settings.
    4. Flagging data points with matrix profile distances outside this range as potential anomalies.

    This approach is robust to extreme values and provides a flexible way to adjust the sensitivity of anomaly detection through configurable percentile thresholds.
    """

    percentiles: Dict[Sensitivities, Tuple[float, float]] = Field(
        {
            # High sensitivity = more anomalies + higher false positives
            # Data point outside of bottom 70% of the MP distances considered anomalous
            "high": [0.3, 0.7],
            # Medium sensitivity = lesser anomalies + lesser false positives
            # Data point outside of bottom 80% of the MP distances considered anomalous
            "medium": [0.2, 0.8],
            # Low sensitivity = least anomalies + least false positives
            # Data point outside of bottom 90% of the MP distances considered anomalous
            "low": [0.1, 0.9],
        },
        description="Lower and upper bounds for high sensitivity",
    )

    iqr_scaling_factor: Dict[Sensitivities, float] = Field(
        {
            "high": 1.5,
            "medium": 1.5,
            "low": 1.5,
        },
        description="Scaling factor for IQR based thresholding",
    )

    @inject
    def batch_score(
        self,
        values: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        time_budget_ms: int | None = None,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> FlagsAndScores:
        """
        Scores anomalies by computing the distance of the relevant MP distance from quartiles. This approach is not swayed by
        extreme values in MP distances. It also converts the score to a flag with a more meaningful interpretation of score.

        The interquartile ranges for scoring are computed using the distances passed in as mp_dist_baseline. If mp_dist_baseline is
        None then the interquartile ranges are computed from mp_dist_to_score

        Parameters:
        values: npt.NDArray[np.float64]
            Array of historical values
        timestamps: npt.NDArray[np.float64]
            Array of timestamps corresponding to historical values
        mp_dist: npt.NDArray[np.float64]
            Array of matrix profile distances for historical values
        ad_config: AnomalyDetectionConfig
            Configuration for anomaly detection
        window_size: int
            Size of the window used for matrix profile computation
        time_budget_ms: int | None = None,
        """
        scores: List[float] = []
        flags: List[AnomalyFlags] = []
        thresholds: List[List[Threshold]] = []
        time_allocated = datetime.timedelta(milliseconds=time_budget_ms) if time_budget_ms else None
        time_start = datetime.datetime.now()
        # Compute score and anomaly flags
        mp_dist_threshold = self._get_mp_dist_threshold(mp_dist, ad_config.sensitivity)
        idx_to_detect_location_from = (
            len(mp_dist) - algo_config.direction_detection_num_timesteps_in_batch_mode
        )
        batch_size = 10 if len(mp_dist) > 10 else 1
        for i, val in enumerate(mp_dist):
            if time_allocated is not None and i % batch_size == 0:
                time_elapsed = datetime.datetime.now() - time_start
                if time_allocated is not None and time_elapsed > time_allocated:
                    sentry_sdk.set_extra("time_taken_for_batch_detection", time_elapsed)
                    sentry_sdk.set_extra("time_allocated_for_batch_detection", time_allocated)
                    sentry_sdk.capture_message(
                        "batch_detection_took_too_long",
                        level="error",
                    )
                    raise ServerError("Batch detection took too long")
            scores.append(0.0 if np.isnan(val) or np.isinf(val) else val - mp_dist_threshold)
            cur_thresholds = [
                Threshold(
                    type=ThresholdType.MP_DIST_IQR, upper=mp_dist_threshold, lower=mp_dist_threshold
                )
            ]

            flag = self._to_flag(val, mp_dist_threshold)
            if i >= idx_to_detect_location_from:
                flag, location_thresholds = self._adjust_flag_for_direction(
                    flag,
                    ad_config.direction,
                    streamed_value=values[i],
                    streamed_timestamp=timestamps[i],
                    history_values=values[0 : i - 1],
                    history_timestamps=timestamps[0 : i - 1],
                    location_detector=location_detector,
                )
                cur_thresholds.extend(location_thresholds)
            flags.append(flag)
            thresholds.append(cur_thresholds)
        return FlagsAndScores(
            flags=flags,
            scores=scores,
            thresholds=thresholds,
        )

    @inject
    def stream_score(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        streamed_mp_dist: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
        history_mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> FlagsAndScores:
        """
        Scores anomalies by computing the distance of the relevant MP distance from quartiles. It also converts the score
        to a flag with a more meaningful interpretation of score.

        The interquartile ranges for scoring are computed using the distances passed in as mp_dist_baseline. If mp_dist_baseline is
        None then the interquartile ranges are computed from mp_dist_to_score

        Parameters:
        streamed_value: np.float64
            The current value being streamed
        streamed_timestamp: np.float64
            The timestamp of the current value being streamed
        streamed_mp_dist: np.float64
            The matrix profile distance for the streamed value
        history_values: npt.NDArray[np.float64]
            Array of historical values
        history_timestamps: npt.NDArray[np.float64]
            Array of timestamps corresponding to historical values
        history_mp_dist: npt.NDArray[np.float64]
            Array of matrix profile distances for historical values
        ad_config: AnomalyDetectionConfig
            Configuration for anomaly detection
        window_size: int
            Size of the window used for matrix profile computation

        Returns:
            FlagsAndScores: Object containing anomaly flags, scores, and thresholds

        """
        mp_dist_threshold = self._get_mp_dist_threshold(history_mp_dist, ad_config.sensitivity)

        # Compute score and anomaly flags
        score = (
            0.0
            if np.isnan(streamed_mp_dist) or np.isinf(streamed_mp_dist)
            else streamed_mp_dist - mp_dist_threshold
        )
        flag = self._to_flag(streamed_mp_dist, mp_dist_threshold)
        flag, thresholds = self._adjust_flag_for_direction(
            flag,
            ad_config.direction,
            streamed_value=streamed_value,
            streamed_timestamp=streamed_timestamp,
            history_values=history_values,
            history_timestamps=history_timestamps,
            location_detector=location_detector,
        )
        thresholds.append(
            Threshold(
                type=ThresholdType.MP_DIST_IQR, upper=mp_dist_threshold, lower=mp_dist_threshold
            )
        )
        return FlagsAndScores(
            flags=[flag],
            scores=[score],
            thresholds=[thresholds],
        )

    def _get_mp_dist_threshold(
        self, mp_dist: npt.NDArray[np.float64], sensitivity: Sensitivities
    ) -> float:
        if sensitivity not in self.percentiles or sensitivity not in self.iqr_scaling_factor:
            raise ClientError(f"Invalid sensitivity: {sensitivity}")

        # Compute the quantiles for threshold level for the sensitivity
        mp_dist_baseline_finite = mp_dist[np.isfinite(mp_dist)]
        median = np.median(mp_dist_baseline_finite)
        variation = np.std(mp_dist_baseline_finite) / median if median > 0 else np.inf

        # Apply additional scaling if variation is low
        scaling = 2.0 if variation < 0.1 else 1.0

        [Q1, Q3] = np.quantile(mp_dist_baseline_finite, self.percentiles[sensitivity])
        IQR = Q3 - Q1

        return Q3 + (scaling * self.iqr_scaling_factor[sensitivity] * IQR)

    def _to_flag(self, mp_dist: np.float64, threshold: float):
        if np.isnan(mp_dist) or mp_dist <= threshold:
            return "none"
        return "anomaly_higher_confidence"

    @inject
    def _adjust_flag_for_direction(
        self,
        flag: AnomalyFlags,
        direction: Directions,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
        location_detector: LocationDetector,
    ) -> Tuple[AnomalyFlags, List[Threshold]]:
        """
        Adjusts the anomaly flag based on the specified direction and time series context.

        Parameters:
        flag: AnomalyFlags
            The current anomaly flag
        direction: Directions
            The direction of the anomaly to detect
        streamed_value: np.float64
            The current value being streamed
        streamed_timestamp: np.float64
            The timestamp of the current value being streamed
        history_values: npt.NDArray[np.float64]
            Array of historical values
        Returns:
        AnomalyFlags
            The adjusted anomaly flag
        List[Threshold]
            The thresholds used for anomaly flag
        """
        if flag == "none" or direction == "both":
            return flag, []

        relative_location = location_detector.detect(
            streamed_value, streamed_timestamp, history_values, history_timestamps
        )
        if relative_location is None:
            return flag, []
        # if direction == "both" and location == PointLocation.NONE:
        #     return "none"
        if (direction == "up" and relative_location.location != PointLocation.UP) or (
            direction == "down" and relative_location.location != PointLocation.DOWN
        ):
            return "none", relative_location.thresholds
        return flag, relative_location.thresholds


class MPCascadingScorer(MPScorer):
    """
    This class implements a cascading scoring mechanism for Matrix Profile-based anomaly detection.
    It applies multiple scorers in sequence, returning the result of the first scorer that produces a valid output.
    This approach allows for fallback strategies and potentially more robust anomaly detection.

    The default implementation uses the LowVarianceScorer and the MPIQRScorer.
    """

    scorers: list[MPScorer] = Field(
        [LowVarianceScorer(), MPIQRScorer()], description="The list of scorers to cascade"
    )

    @inject
    def batch_score(
        self,
        values: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        time_budget_ms: int | None = None,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> Optional[FlagsAndScores]:
        for scorer in self.scorers:
            flags_and_scores = scorer.batch_score(
                values,
                timestamps,
                mp_dist,
                ad_config,
                window_size,
                time_budget_ms,
                algo_config,
                location_detector,
            )
            if flags_and_scores is not None:
                return flags_and_scores
        return None

    @inject
    def stream_score(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        streamed_mp_dist: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
        history_mp_dist: npt.NDArray[np.float64],
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
        location_detector: LocationDetector = injected,
    ) -> Optional[FlagsAndScores]:
        for scorer in self.scorers:
            flags_and_scores = scorer.stream_score(
                streamed_value,
                streamed_timestamp,
                streamed_mp_dist,
                history_values,
                history_timestamps,
                history_mp_dist,
                ad_config,
                window_size,
                algo_config,
                location_detector,
            )
            if flags_and_scores is not None:
                return flags_and_scores
        return None
