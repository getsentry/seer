import datetime
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import sentry_sdk
from pydantic import Field

from seer.anomaly_detection.detectors.location_detectors import LocationDetector
from seer.anomaly_detection.detectors.mp_scorers import FlagsAndScores, MPScorer
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


class MPBoxCoxScorer(MPScorer):
    """
    This class implements a scoring method for detecting anomalies in time series data using the Box-Cox transformation.
    The Box-Cox transformation is applied to normalize the data, followed by z-score based anomaly detection.
    """

    box_cox_lambda: float = Field(
        0.0,
        description="The lambda parameter for the Box-Cox transformation. Default 0 corresponds to log transform.",
    )
    z_score_thresholds: Dict[Sensitivities, float] = Field(
        {
            "high": 2.0,  # 95.4% confidence interval
            "medium": 2.5,  # 98.8% confidence interval
            "low": 3.0,  # 99.7% confidence interval
        },
        description="Z-score thresholds for different sensitivity levels",
    )

    def _inverse_box_cox_transform(self, x: float, bc_lambda: float) -> float:
        """Apply inverse Box-Cox transformation to return data to original scale.

        Args:
            x: The Box-Cox transformed value
            bc_lambda: The lambda parameter for the Box-Cox transformation

        Returns:
            float: The inverse transformed value in the original scale
        """
        if bc_lambda == 0:
            return np.exp([x])[0] - 1
        return np.power(bc_lambda * x + 1, 1 / bc_lambda) - 1

    # def _inverse_box_cox_transform(self, x: float, bc_lambda: float) -> float:
    #     """Apply inverse Box-Cox transformation to return data to original scale.

    #     Parameters:
    #         x: The Box-Cox transformed data
    #         bc_lambda: The lambda parameter for the Box-Cox transformation

    #     Returns:
    #         The inverse transformed data in the original scale
    #     """
    #     if bc_lambda <= 0:
    #         return np.exp([x])[0] - 1
    #     return special.inv_boxcox([x], bc_lambda)[0] - 1

    def _box_cox_transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply Box-Cox transformation to the data."""
        # Ensure data is positive for Box-Cox transform
        min_val = x.min()
        if min_val <= 0:
            x = x - min_val + 1

        if self.box_cox_lambda == 0:
            return np.log(x)
        return (np.power(x, self.box_cox_lambda) - 1) / self.box_cox_lambda

    def _get_z_scores(
        self, values: npt.NDArray[np.float64], sensitivity: Sensitivities
    ) -> Tuple[npt.NDArray[np.float64], float, float, float]:
        """Calculate z-scores and threshold."""
        if sensitivity not in self.z_score_thresholds:
            raise ClientError(f"Invalid sensitivity: {sensitivity}")

        transformed = self._box_cox_transform(values)
        mean = np.mean(transformed)
        std = float(np.std(transformed))
        z_scores = (transformed - mean) / std if std > 0 else np.zeros_like(transformed)
        threshold = self.z_score_thresholds[sensitivity]
        threshold_transformed = self._inverse_box_cox_transform(threshold, self.box_cox_lambda)

        return z_scores, threshold, std, threshold_transformed

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
        z_scores, threshold, std, threshold_transformed = self._get_z_scores(
            values, ad_config.sensitivity
        )

        scores = []
        flags = []
        thresholds = []
        time_allocated = datetime.timedelta(milliseconds=time_budget_ms) if time_budget_ms else None
        time_start = datetime.datetime.now()

        idx_to_detect_location_from = (
            len(values) - algo_config.direction_detection_num_timesteps_in_batch_mode
        )
        batch_size = 10 if len(values) > 10 else 1
        for i, z_score in enumerate(z_scores):
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
            score = z_score
            flag: AnomalyFlags = "none"
            location_thresholds: List[Threshold] = []
            if std != 0 and score > threshold:
                flag = "anomaly_higher_confidence"
                if i >= idx_to_detect_location_from:
                    flag, location_thresholds = self._adjust_flag_for_direction(
                        flag,
                        ad_config.direction,
                        values[i],
                        timestamps[i],
                        values[:i],
                        timestamps[:i],
                        location_detector,
                    )
            cur_thresholds = [
                Threshold(
                    type=ThresholdType.BOX_COX_THRESHOLD,
                    upper=threshold_transformed,
                    lower=-threshold_transformed,
                )
            ]

            scores.append(score)
            flags.append(flag)
            cur_thresholds.extend(location_thresholds)
            thresholds.append(cur_thresholds)

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
    ) -> FlagsAndScores:
        # Include current value in z-score calculation
        values = np.append(history_values, streamed_value)
        z_scores, threshold, std, threshold_transformed = self._get_z_scores(
            values, ad_config.sensitivity
        )

        # Get z-score for streamed value
        score = z_scores[-1]
        thresholds: List[Threshold] = []
        flag = "none"

        if std != 0 and score > threshold:
            flag, thresholds = self._adjust_flag_for_direction(
                "anomaly_higher_confidence",
                ad_config.direction,
                streamed_value,
                streamed_timestamp,
                history_values,
                history_timestamps,
                location_detector,
            )

        thresholds.append(
            Threshold(
                type=ThresholdType.BOX_COX_THRESHOLD,
                upper=threshold_transformed,
                lower=threshold_transformed,
            )
        )

        return FlagsAndScores(
            flags=[flag],
            scores=[score],
            thresholds=[thresholds],
        )

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
        if flag == "none" or direction == "both":
            return flag, []

        if len(history_values) == 0:
            raise ValueError("No history values to detect location")

        relative_location = location_detector.detect(
            streamed_value, streamed_timestamp, history_values, history_timestamps
        )
        if relative_location is None:
            return flag, []

        if (direction == "up" and relative_location.location != PointLocation.UP) or (
            direction == "down" and relative_location.location != PointLocation.DOWN
        ):
            return "none", relative_location.thresholds
        return flag, relative_location.thresholds
