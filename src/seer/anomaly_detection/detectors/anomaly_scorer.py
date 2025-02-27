import abc
import logging

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
from seer.anomaly_detection.models import (
    AlgoConfig,
    AnomalyDetectionConfig,
    AnomalyFlags,
    Directions,
)
from seer.dependency_injection import inject, injected
from seer.exceptions import ServerError

logger = logging.getLogger(__name__)


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
            ad_config=ad_config,
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
            # logger.warning("The prophet_df is None or empty, skipping prophet scoring")
            return mp_flags_and_scores

        # Lookup row in prophet_df for streamed_timestam and update y and actual with the new streamed value
        row_exists = (prophet_df["ds"] == streamed_timestamp).any()
        if not row_exists:
            logger.warning(
                "Row for timestamp not found in prophet_df, skipping prophet scoring",
                extra={"streamed_timestamp": streamed_timestamp},
            )
            return mp_flags_and_scores
        prophet_df.loc[prophet_df.ds == streamed_timestamp, "y"] = float(streamed_value)
        prophet_df.loc[prophet_df.ds == streamed_timestamp, "actual"] = float(streamed_value)
        df_prophet_scores = self.prophet_scorer.batch_score(prophet_df)

        combined = self._merge_prophet_mp_results(
            timestamps=np.array([np.float64(streamed_timestamp)]),
            mp_flags_and_scores=mp_flags_and_scores,
            prophet_predictions=df_prophet_scores,
            ad_config=ad_config,
        )
        return combined

    def _adjust_prophet_flag_for_location(
        self,
        mp_flag: AnomalyFlags,
        prev_mp_flag: AnomalyFlags,
        prophet_flag: AnomalyFlags,
        y: np.float64,
        yhat: np.float64,
        yhat_lower: np.float64,
        yhat_upper: np.float64,
        direction: Directions,
    ) -> AnomalyFlags:
        if (
            direction == "both"
            or mp_flag == "none"
            or mp_flag == prev_mp_flag
            or prophet_flag == "none"
        ):
            return prophet_flag

        if (direction == "up" and y >= yhat_upper) or (direction == "down" and y <= yhat_lower):
            return "anomaly_higher_confidence"
        else:
            return "none"

    def _merge_prophet_mp_results(
        self,
        timestamps: np.ndarray,
        mp_flags_and_scores: FlagsAndScores,
        prophet_predictions: pd.DataFrame,
        ad_config: AnomalyDetectionConfig,
    ) -> FlagsAndScores:

        # todo: return prophet thresholds
        def merge(timestamps, mp_flags, prophet_map):
            flags = []
            found = 0
            previous_mp_flag: AnomalyFlags = "none"
            for timestamp, mp_flag in zip(timestamps, mp_flags):
                if pd.to_datetime(timestamp) in prophet_map["flag"]:
                    found += 1
                    prophet_flag = prophet_map["flag"][pd.to_datetime(timestamp)]
                    prophet_score = prophet_map["score"][pd.to_datetime(timestamp)]
                    prophet_flag = self._adjust_prophet_flag_for_location(
                        mp_flag=mp_flag,
                        prev_mp_flag=previous_mp_flag,
                        prophet_flag=prophet_flag,
                        y=prophet_map["y"],
                        yhat=prophet_map["yhat"],
                        yhat_lower=prophet_map["yhat_lower"],
                        yhat_upper=prophet_map["yhat_upper"],
                        direction=ad_config.direction,
                    )
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
                previous_mp_flag = mp_flag
            # todo: publish metrics for found/total
            # if debug:
            #     print(f"found {found} out of {len(timestamps)}")
            return flags

        prophet_predictions_map = prophet_predictions.set_index("ds")[
            ["flag", "score", "y", "yhat", "yhat_lower", "yhat_upper"]
        ].to_dict()

        flags = merge(timestamps, mp_flags_and_scores.flags, prophet_predictions_map)

        return FlagsAndScores(
            flags=flags,
            scores=mp_flags_and_scores.scores,
            thresholds=mp_flags_and_scores.thresholds,
        )
