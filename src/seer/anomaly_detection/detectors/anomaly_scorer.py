import abc

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field

from seer.anomaly_detection.detectors.mp_cascading_scorer import MPCascadingScorer
from seer.anomaly_detection.detectors.mp_scorers import FlagsAndScores, MPScorer
from seer.anomaly_detection.detectors.prophet_scorer import (
    ProphetScaledSmoothedScorer,
    ProphetScorer,
)
from seer.anomaly_detection.models import AlgoConfig, AnomalyDetectionConfig
from seer.dependency_injection import inject, injected
from seer.exceptions import ServerError


class AnomalyScorer(BaseModel, abc.ABC):
    @abc.abstractmethod
    def batch_score(
        self,
        values: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        prophet_df: pd.DataFrame | None,
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
    ) -> FlagsAndScores:
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
        prophet_df: pd.DataFrame | None,
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
    ) -> FlagsAndScores:
        return NotImplemented


class CombinedAnomalyScorer(AnomalyScorer):
    """
    A scorer that combines multiple scoring methods to detect anomalies.
    The combined approach helps reduce false positives while maintaining sensitivity
    to different types of anomalies.
    """

    mp_scorer: MPScorer = Field(
        default=MPCascadingScorer(),
        description="The MPScorer to use for scoring the matrix profile distance",
    )

    prophet_scorer: ProphetScorer = Field(
        default=ProphetScaledSmoothedScorer(),
        description="The ProphetScorer to use for scoring against the prophet model",
    )

    @inject
    def batch_score(
        self,
        values: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        prophet_df: pd.DataFrame | None,
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
    ) -> FlagsAndScores:
        """
        Score anomalies by combining MP and Prophet scores.

        Args:
            values: Array of time series values
            timestamps: Array of timestamps
            mp_dist: Matrix profile distances
            prophet_df: Prophet model dataframe
            ad_config: Anomaly detection configuration
            window_size: Size of the sliding window
            time_budget_ms: Optional time budget in milliseconds
            algo_config: Algorithm configuration

        Returns:
            Combined flags and scores from MP and Prophet scoring methods
        """

        mp_flags_and_scores = self.mp_scorer.batch_score(
            values=values,
            timestamps=timestamps,
            mp_dist=mp_dist,
            ad_config=ad_config,
            window_size=window_size,
            algo_config=algo_config,
        )
        if mp_flags_and_scores is None:
            raise ServerError("No flags and scores from MP scorer")

        if prophet_df is None or prophet_df.empty:
            return mp_flags_and_scores

        df_prophet_scores = self.prophet_scorer.batch_score(prophet_df)
        combined = self._merge_prophet_mp_results(
            timestamps=timestamps,
            mp_flags_and_scores=mp_flags_and_scores,
            prophet_predictions=df_prophet_scores,
        )
        return combined

    @inject
    def stream_score(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        streamed_mp_dist: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
        history_mp_dist: npt.NDArray[np.float64],
        prophet_df: pd.DataFrame | None,
        ad_config: AnomalyDetectionConfig,
        window_size: int,
        algo_config: AlgoConfig = injected,
    ) -> FlagsAndScores:
        mp_flags_and_scores = self.mp_scorer.stream_score(
            streamed_value=streamed_value,
            streamed_timestamp=streamed_timestamp,
            streamed_mp_dist=streamed_mp_dist,
            history_values=history_values,
            history_timestamps=history_timestamps,
            history_mp_dist=history_mp_dist,
            ad_config=ad_config,
            window_size=window_size,
            algo_config=algo_config,
        )
        if mp_flags_and_scores is None:
            raise ServerError("No flags and scores from MP scorer")

        if prophet_df is None or prophet_df.empty:
            return mp_flags_and_scores

        return NotImplemented  # TODO: implement combined scoring for streaming

    def _merge_prophet_mp_results(
        self,
        timestamps: np.ndarray,
        mp_flags_and_scores: FlagsAndScores,
        prophet_predictions: pd.DataFrame,
    ) -> FlagsAndScores:

        # todo: return prophet thresholds
        # todo: apply location logic using prophet_predictions

        def merge(timestamps, mp_flags, prophet_map):
            flags = []
            found = 0
            for timestamp, mp_flag in zip(timestamps, mp_flags):
                if pd.to_datetime(timestamp) in prophet_map["flag"]:
                    found += 1
                    prophet_flag = prophet_map["flag"][pd.to_datetime(timestamp)]
                    prophet_score = prophet_map["score"][pd.to_datetime(timestamp)]
                    if (
                        mp_flag == "anomaly_higher_confidence"
                        and prophet_flag == "anomaly_higher_confidence"
                    ):
                        flags.append("anomaly_higher_confidence")
                    elif prophet_score >= 2.0:
                        flags.append(prophet_flag)
                    else:
                        flags.append("none")
                else:
                    flags.append(mp_flag)
            # todo: publish metrics for found/total
            # if debug:
            #     print(f"found {found} out of {len(timestamps)}")
            return flags

        prophet_predictions_map = prophet_predictions.set_index("ds")[["flag", "score"]].to_dict()

        flags = merge(timestamps, mp_flags_and_scores.flags, prophet_predictions_map)

        return FlagsAndScores(
            flags=flags,
            scores=mp_flags_and_scores.scores,
            thresholds=mp_flags_and_scores.thresholds,
        )

    # todo: fix me and apply location logic using prophet_predictions
    # def _adjust_flag_for_direction(
    #     self,
    #     flag: AnomalyFlags,
    #     direction: Directions,
    #     streamed_value: np.float64,
    #     streamed_timestamp: np.float64,
    #     history_values: npt.NDArray[np.float64],
    #     history_timestamps: npt.NDArray[np.float64],
    #     location_detector: LocationDetector,
    # ) -> Tuple[AnomalyFlags, List[Threshold]]:
    #     if flag == "none" or direction == "both":
    #         return flag, []

    #     if len(history_values) == 0:
    #         raise ValueError("No history values to detect location")
    #     relative_location = location_detector.detect(
    #         streamed_value, streamed_timestamp, history_values, history_timestamps
    #     )
    #     if relative_location is None:
    #         return flag, []

    #     if (direction == "up" and relative_location.location != PointLocation.UP) or (
    #         direction == "down" and relative_location.location != PointLocation.DOWN
    #     ):
    #         return "none", relative_location.thresholds
    #     return flag, relative_location.thresholds

    # def _detect_location(
    #     self,
    #     streamed_value: np.float64,
    #     streamed_timestamp: np.float64,
    #     forecast: pd.DataFrame,
    #     algo_config: AlgoConfig,
    # ) -> Tuple[PointLocation, List[Threshold]]:
    #     if algo_config.prophet_uncertainty_samples > 0 or algo_config.prophet_mcmc_samples > 0:
    #         streamed_forecast = forecast.loc[len(forecast) - 1]
    #         yhat_upper = streamed_forecast["yhat_upper"]
    #         yhat_lower = streamed_forecast["yhat_lower"]
    #         if algo_config.return_thresholds:
    #             thresholds = [
    #                 Threshold(
    #                     type=ThresholdType.PREDICTION,
    #                     timestamp=streamed_timestamp,
    #                     upper=yhat_upper,
    #                     lower=yhat_lower,
    #                 ),
    #                 Threshold(
    #                     type=ThresholdType.TREND,
    #                     timestamp=streamed_timestamp,
    #                     upper=streamed_forecast["trend_upper"],
    #                     lower=streamed_forecast["trend_lower"],
    #                 ),
    #             ]
    #         else:
    #             thresholds = []

    #         if streamed_value > yhat_upper:
    #             return RelativeLocation(
    #                 location=PointLocation.UP,
    #                 thresholds=thresholds,
    #             )
    #         elif streamed_value < yhat_lower:
    #             return RelativeLocation(
    #                 location=PointLocation.DOWN,
    #                 thresholds=thresholds,
    #             )
    #         else:
    #             return RelativeLocation(
    #                 location=PointLocation.NONE,
    #                 thresholds=thresholds,
    #             )
    #     else:
    #         forecast = forecast.yhat.loc[len(forecast) - 1]
    #         if np.isclose(streamed_value, forecast, rtol=1e-5, atol=1e-8):
    #             return RelativeLocation(
    #                 location=PointLocation.NONE,
    #                 thresholds=[],
    #             )
    #         elif streamed_value > forecast:
    #             return RelativeLocation(
    #                 location=PointLocation.UP,
    #                 thresholds=[],
    #             )
    #         else:
    #             return RelativeLocation(
    #                 location=PointLocation.DOWN,
    #                 thresholds=[],
    #             )
