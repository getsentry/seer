import abc
import logging
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import sentry_sdk
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.models import (
    AlgoConfig,
    AnomalyDetectionConfig,
    AnomalyFlags,
    Sensitivities,
    Threshold,
    ThresholdType,
)
from seer.dependency_injection import inject, injected
from seer.exceptions import ClientError
from seer.tags import AnomalyDetectionTags

logger = logging.getLogger(__name__)


class FlagsAndScores(BaseModel):
    flags: List[AnomalyFlags]
    scores: List[float]
    thresholds: List[List[Threshold]]
    confidence_levels: List[str]

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
    ) -> Optional[FlagsAndScores]:
        return NotImplemented


class MPLowVarianceScorer(MPScorer):
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
    ) -> Optional[FlagsAndScores]:
        ts_mean = values.mean()
        scores = []
        flags = []
        thresholds = []
        confidence_levels = []
        if values.std() > self.std_threshold:
            sentry_sdk.set_tag(AnomalyDetectionTags.LOW_VARIATION_TS, 0)
            return None

        sentry_sdk.set_tag(AnomalyDetectionTags.LOW_VARIATION_TS, 1)
        for i, val in enumerate(values):
            flag, score, pred_up, pred_down = self._to_flag_and_score(val, ts_mean, ad_config)
            flags.append(flag)
            scores.append(score)
            thresholds.append(
                [
                    Threshold(
                        type=ThresholdType.LOW_VARIANCE_THRESHOLD,
                        timestamp=timestamps[i],
                        upper=pred_up,
                        lower=pred_down,
                    )
                ]
            )
            confidence_levels.append("medium")  # Default to medium confidence for low variance
        return FlagsAndScores(
            flags=flags, scores=scores, thresholds=thresholds, confidence_levels=confidence_levels
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
    ) -> Optional[FlagsAndScores]:
        context = history_values[-2 * window_size :]
        if context.std() > self.std_threshold:
            return None
        flag, score, pred_up, pred_down = self._to_flag_and_score(
            streamed_value, context.mean(), ad_config
        )
        threshold = Threshold(
            type=ThresholdType.LOW_VARIANCE_THRESHOLD,
            timestamp=streamed_timestamp,
            upper=pred_up,
            lower=pred_down,
        )

        return FlagsAndScores(
            flags=[flag],
            scores=[score],
            thresholds=[[threshold]],
            confidence_levels=["medium"],  # Default to medium for low variance
        )
