import abc
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import sentry_sdk
from pydantic import BaseModel, Field

from seer.anomaly_detection.models import AnomalyFlags, Directions, Sensitivities
from seer.exceptions import ClientError
from seer.tags import AnomalyDetectionTags

logger = logging.getLogger(__name__)


class FlagsAndScores(BaseModel):
    flags: list[AnomalyFlags]
    scores: list[float]


class MPScorer(BaseModel, abc.ABC):
    """
    Abstract base class for calculating an anomaly score
    """

    @abc.abstractmethod
    def batch_score(
        self,
        ts: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> Optional[FlagsAndScores]:
        return NotImplemented

    @abc.abstractmethod
    def stream_score(
        self,
        ts_streamed: np.float64,
        mp_dist_streamed: np.float64,
        ts_history: npt.NDArray[np.float64],
        mp_dist_history: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> Optional[FlagsAndScores]:
        return NotImplemented


class LowVarianceScorer(MPScorer):
    """
    This class scores anomalies using mean if the series has a low variance.
    """

    std_threshold: float = Field(
        0.01,
        description="Minimum standard deviation required in order to use IQR based scoring",
    )
    scaling_factors: Dict[Sensitivities, int] = Field(
        {
            # High sensitivity = more anomalies + higher false positives
            # 2x the mean and above is considered an anomaly
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
        sensitivity: Sensitivities,
        direction: Directions,
    ) -> tuple[AnomalyFlags, float]:
        # High sensitivity will mark more values as anomalies
        if sensitivity not in self.scaling_factors:
            raise ClientError(f"Invalid sensitivity: {sensitivity}")
        scaling_factor = self.scaling_factors[sensitivity]
        bound1 = ts_mean + ts_mean * scaling_factor
        bound2 = ts_mean - ts_mean * scaling_factor
        lower_bound = min(bound1, bound2)
        upper_bound = max(bound1, bound2)

        # if current value is significantly higher or lower than the mean then mark it as high anomaly else mark it as no anomaly
        if val < lower_bound or val > upper_bound:
            return "anomaly_higher_confidence", 0.9
        return "none", 0.0

    def batch_score(
        self,
        ts: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> Optional[FlagsAndScores]:
        ts_mean = ts.mean()
        scores = []
        flags = []
        if ts.std() > self.std_threshold:
            sentry_sdk.set_tag(AnomalyDetectionTags.LOW_VARIATION_TS, 0)
            return None

        sentry_sdk.set_tag(AnomalyDetectionTags.LOW_VARIATION_TS, 1)
        for val in ts:
            flag, score = self._to_flag_and_score(
                val, ts_mean, sensitivity=sensitivity, direction=direction
            )
            flags.append(flag)
            scores.append(score)
        return FlagsAndScores(flags=flags, scores=scores)

    def stream_score(
        self,
        ts_streamed: np.float64,
        mp_dist_streamed: np.float64,
        ts_history: npt.NDArray[np.float64],
        mp_dist_history: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> Optional[FlagsAndScores]:
        context = ts_history[-2 * window_size :]
        if context.std() > self.std_threshold:
            return None
        flag, score = self._to_flag_and_score(
            ts_streamed, context.mean(), sensitivity=sensitivity, direction=direction
        )

        return FlagsAndScores(flags=[flag], scores=[score])


class MPIQRScorer(MPScorer):
    """
    This class scores anomalies using the interquartile range of the matrix profile distances.
    """

    percentiles: Dict[Sensitivities, Tuple[float, float]] = Field(
        {
            # High sensitivity = more anomalies + higher false positives
            # Data point outside of bottom 65% of the MP distances considered anomalous
            "high": [0.25, 0.75],
            # Medium sensitivity = lesser anomalies + lesser false positives
            # Data point outside of bottom 70% of the MP distances considered anomalous
            "medium": [0.15, 0.85],
            # Low sensitivity = leaset anomalies + leaset false positives
            # Data point outside of bottom 90% of the MP distances considered anomalous
            "low": [0.05, 0.95],
        },
        description="Lower and upper bounds for high sensitivity",
    )

    def batch_score(
        self,
        ts: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> FlagsAndScores:
        """
        Scores anomalies by computing the distance of the relevant MP distance from quartiles. This approach is not swayed by
        extreme values in MP distances. It also converts the score to a flag with a more meaningful interpretation of score.

        The interquartile ranges for scoring are computed using the distances passed in as mp_dist_baseline. If mp_dist_baseline is
        None then the interquartile ranges are computed from mp_dist_to_score

        Parameters:
        mp_dist_to_score: Numpy array
            The matrix profile distances that need scoring
        sensitivity: Sensitivities
            Low sensitivity will detect more anomalies with more false positives and high sensitiviy will detect less anomalies with more false negatives
        direction: Directions
            Up will detect anomaly only if the detected anomaly is in the upward direction while Down will detect it only if it is downward. Both will cover both up and down.

        Returns:
            tuple with list of scores and list of flags, where each flag is one of
            * "none" - indicating not an anomaly
            * "anomaly_lower_confidence" - indicating anomaly but only with a lower threshold
            * "anomaly_higher_confidence" - indicating anomaly with a higher threshold
        """
        scores: list[float] = []
        flags: list[AnomalyFlags] = []
        # Compute score and anomaly flags
        threshold = self._get_threshold(mp_dist, sensitivity)
        for i, val in enumerate(mp_dist):
            scores.append(0.0 if np.isnan(val) or np.isinf(val) else val - threshold)
            flag = self._to_flag(val, threshold)
            if i > 2 * window_size:
                flag = self._adjust_flag_for_vicinity(
                    flag=flag, ts_value=ts[i], context=ts[i - 2 * window_size : i - 1]
                )
            flags.append(flag)

        return FlagsAndScores(flags=flags, scores=scores)

    def stream_score(
        self,
        ts_streamed: np.float64,
        mp_dist_streamed: np.float64,
        ts_history: npt.NDArray[np.float64],
        mp_dist_history: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> FlagsAndScores:
        """
        Scores anomalies by computing the distance of the relevant MP distance from quartiles. This approach is not swayed by
        extreme values in MP distances. It also converts the score to a flag with a more meaningful interpretation of score.

        The interquartile ranges for scoring are computed using the distances passed in as mp_dist_baseline. If mp_dist_baseline is
        None then the interquartile ranges are computed from mp_dist_to_score

        Parameters:
        mp_dist_to_score: Numpy array
            The matrix profile distances that need scoring
        sensitivity: Sensitivities
            Low sensitivity will detect more anomalies with more false positives and high sensitiviy will detect less anomalies with more false negatives
        direction: Directions
            Up will detect anomaly only if the detected anomaly is in the upward direction while Down will detect it only if it is downward. Both will cover both up and down.
        mp_dist_baseline: Numpy array
            Baseline distances used for calculating inter quartile range

        Returns:
            tuple with list of scores and list of flags, where each flag is one of
            * "none" - indicating not an anomaly
            * "anomaly_lower_confidence" - indicating anomaly but only with a lower threshold
            * "anomaly_higher_confidence" - indicating anomaly with a higher threshold
        """
        threshold = self._get_threshold(mp_dist_history, sensitivity)

        # Compute score and anomaly flags
        score = (
            0.0
            if np.isnan(mp_dist_streamed) or np.isinf(mp_dist_streamed)
            else mp_dist_streamed - threshold
        )
        flag = self._to_flag(mp_dist_streamed, threshold)
        # anomaly identified. apply logic to check for peak and trough
        flag = self._adjust_flag_for_vicinity(ts_streamed, flag, ts_history[-2 * window_size :])

        return FlagsAndScores(flags=[flag], scores=[score])

    def _get_threshold(self, mp_dist: npt.NDArray[np.float64], sensitivity: Sensitivities) -> float:
        if sensitivity not in self.percentiles:
            raise ClientError(f"Invalid sensitivity: {sensitivity}")

        # Compute the quantiles for threshold level for the sensitivity
        mp_dist_baseline_finite = mp_dist[np.isfinite(mp_dist)]
        [Q1, Q3] = np.quantile(mp_dist_baseline_finite, self.percentiles[sensitivity])
        IQR = Q3 - Q1
        threshold = Q3 + (1.5 * IQR)
        return threshold

    def _to_flag(self, mp_dist: np.float64, threshold: float):
        if np.isnan(mp_dist) or mp_dist <= threshold:
            return "none"
        return "anomaly_higher_confidence"

    def _adjust_flag_for_vicinity(
        self, ts_value: np.float64, flag: AnomalyFlags, context: npt.NDArray[np.float64]
    ) -> AnomalyFlags:
        """
        This method adjusts the severity of a detected anomaly based on the underlying time step's proximity to peaks and troughs.
        The intuition is that for our alerting and metrics use case, an anomaly near peak or trough is more critical than one that is not.
        Current approach is to use the inter quartile range from a subsequence of the time series, identified by the context parameter.

        Parameters:
        ts_value: float
            The time step being analyzied for anomaly

        flag: AnomalyFlags
            Anomaly identified before applying this peak-trough logic

        context: npt.NDArray[np.float64]
            Time series subsequence that is used for teak-trough detection

        """
        # if flag == "anomaly_higher_confidence" or flag == "anomaly_lower_confidence":
        #     [Q1, Q3] = np.quantile(context, [0.25, 0.75])
        #     IQR = Q3 - Q1
        #     threshold_lower = Q1 - (0 * IQR)
        #     threshold_upper = Q3 + (0 * IQR)
        #     # if ts_value > Q1 and ts_value < Q3:
        #     if ts_value >= threshold_lower and ts_value <= threshold_upper:
        #         flag = "anomaly_lower_confidence"
        #     else:
        #         flag = "anomaly_higher_confidence"
        return flag


class MPCascadingScorer(MPScorer):
    """
    This class combines the results of the LowVarianceScorer and the MPIQRScorer.
    """

    scorers: list[MPScorer] = Field(
        [LowVarianceScorer(), MPIQRScorer()], description="The list of scorers to cascade"
    )

    def batch_score(
        self,
        ts: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> Optional[FlagsAndScores]:
        for scorer in self.scorers:
            flags_and_scores = scorer.batch_score(ts, mp_dist, sensitivity, direction, window_size)
            if flags_and_scores is not None:
                return flags_and_scores
        return None

    def stream_score(
        self,
        ts_streamed: np.float64,
        mp_dist_streamed: np.float64,
        ts_history: npt.NDArray[np.float64],
        mp_dist_history: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> Optional[FlagsAndScores]:
        for scorer in self.scorers:
            flags_and_scores = scorer.stream_score(
                ts_streamed,
                mp_dist_streamed,
                ts_history,
                mp_dist_history,
                sensitivity,
                direction,
                window_size,
            )
            if flags_and_scores is not None:
                return flags_and_scores
        return None
