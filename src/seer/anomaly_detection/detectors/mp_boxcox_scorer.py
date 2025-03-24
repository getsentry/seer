import logging
import sys
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy import special, stats

from seer.anomaly_detection.detectors.mp_scorers import FlagsAndScores, MPScorer
from seer.anomaly_detection.models import (
    AlgoConfig,
    AnomalyDetectionConfig,
    AnomalyFlags,
    ConfidenceLevel,
    Sensitivities,
    Threshold,
    ThresholdType,
)
from seer.dependency_injection import inject, injected
from seer.exceptions import ClientError

logger = logging.getLogger(__name__)


class MPBoxCoxScorer(MPScorer):
    """
    This class implements a scoring method for detecting anomalies in time series data using the Box-Cox transformation.
    The Box-Cox transformation is applied to normalize the data, followed by z-score based anomaly detection.
    """

    z_score_thresholds: Dict[Sensitivities, float] = Field(
        {
            "high": 1.28,  # 90% confidence interval
            "medium": 1.64,  # 95% confidence interval
            "low": 2.32,  # 99% confidence interval
        },
        description="Z-score thresholds for different sensitivity levels",
    )

    def _inverse_box_cox_transform(self, x: float, bc_lambda: float, min_val: float) -> float:
        """Apply inverse Box-Cox transformation to return data to original scale.

        Parameters:
            x: The Box-Cox transformed data
            bc_lambda: The lambda parameter for the Box-Cox transformation

        Returns:
            The inverse transformed data in the original scale
        """

        if bc_lambda <= 0:
            converted = np.exp([x])[0]
        else:
            converted = special.inv_boxcox([x], bc_lambda)[0]
        if min_val <= 0:
            return converted + min_val - 1
        else:
            return converted

    def _box_cox_transform(
        self, x: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], float, float]:
        """Apply Box-Cox transformation to the data.

        Parameters:
            x: The data to be transformed

        Returns:
            The Box-Cox transformed data
        """
        # Get indices of nan values to restore them later
        nan_indices = np.isnan(x)
        nan_count = np.sum(nan_indices)
        x_clean = x[~nan_indices]

        min_val = np.min(x_clean)

        if min_val <= 0:
            x_positive = x_clean - min_val + 1
        else:
            x_positive = x_clean

        # Don't transform if values are constant
        if np.all(x == x[0]):
            transformed = x
            bc_lambda = 0
        else:
            transformed, bc_lambda = stats.boxcox(x_positive)
            if bc_lambda <= 0:
                transformed = np.log(x_positive)

        # Add nan values back to front of array
        if nan_count > 0:
            transformed = np.concatenate([np.full(nan_count, np.nan), transformed])

        return transformed, bc_lambda, min_val

    def _get_z_scores(
        self, values: npt.NDArray[np.float64], sensitivity: Sensitivities
    ) -> Tuple[npt.NDArray[np.float64], float, float, float]:
        """Calculate z-scores and threshold."""
        if sensitivity not in self.z_score_thresholds:
            raise ClientError(f"Invalid sensitivity: {sensitivity}")

        # Get indices of nan values to restore them later
        nan_indices = np.isnan(values)
        values_no_nan = values[~nan_indices]

        transformed, bc_lambda, min_val = self._box_cox_transform(values_no_nan)
        mean = float(np.mean(transformed))
        std = float(np.std(transformed))
        z_scores = (transformed - mean) / std if std > 0 else np.zeros_like(transformed)

        threshold = self.z_score_thresholds[sensitivity]
        threshold_transformed = self._inverse_box_cox_transform(
            (threshold * std) + mean, bc_lambda, min_val
        )

        # Add nans back in the same positions
        z_scores_with_nans = np.empty(len(values))
        z_scores_with_nans[~nan_indices] = z_scores
        z_scores_with_nans[nan_indices] = np.nan

        return z_scores_with_nans, threshold, std, threshold_transformed

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
    ) -> FlagsAndScores:
        z_scores, threshold, std, threshold_transformed = self._get_z_scores(
            mp_dist, ad_config.sensitivity
        )
        scores = []
        flags = []
        thresholds = []
        confidence_levels = []
        for i, score in enumerate(z_scores):
            flag: AnomalyFlags = "none"
            location_thresholds: List[Threshold] = []

            if std != 0 and not np.isnan(score) and score > threshold:
                flag = "anomaly_higher_confidence"

            confidence_level = (
                ConfidenceLevel.HIGH if score >= threshold * 2 else ConfidenceLevel.MEDIUM
            )
            cur_thresholds = [
                Threshold(
                    type=ThresholdType.BOX_COX_THRESHOLD,
                    timestamp=timestamps[i],
                    upper=threshold_transformed,
                    lower=-threshold_transformed,
                )
            ]

            scores.append(
                -sys.float_info.max if np.isnan(score) else score
            )  # Scores are not used and storing NaNs cause redash issues. So storing lowest float value.
            flags.append(flag)
            cur_thresholds.extend(location_thresholds)
            thresholds.append(cur_thresholds)
            confidence_levels.append(confidence_level)
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
    ) -> FlagsAndScores:
        # Include current value in z-score calculation
        values = np.append(history_mp_dist, streamed_mp_dist)
        z_scores, threshold, std, threshold_transformed = self._get_z_scores(
            values[len(values) // 2 :], ad_config.sensitivity
        )

        # Get z-score for streamed value
        score = z_scores[-1]

        flag: AnomalyFlags = (
            "none"
            if std == 0 or np.isnan(score) or score <= threshold
            else "anomaly_higher_confidence"
        )
        confidence_level = (
            ConfidenceLevel.HIGH if score >= threshold * 2 else ConfidenceLevel.MEDIUM
        )
        thresholds: List[Threshold] = [
            Threshold(
                type=ThresholdType.BOX_COX_THRESHOLD,
                timestamp=streamed_timestamp,
                upper=threshold_transformed,
                lower=-threshold_transformed,
            )
        ]
        score = -sys.float_info.max if np.isnan(score) else score

        return FlagsAndScores(
            flags=[flag],
            scores=[score],
            thresholds=[thresholds],
            confidence_levels=[confidence_level],
        )
