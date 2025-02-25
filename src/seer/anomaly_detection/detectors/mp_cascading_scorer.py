from typing import Optional

import numpy as np
import numpy.typing as npt
from pydantic import Field

from seer.anomaly_detection.detectors.mp_boxcox_scorer import MPBoxCoxScorer
from seer.anomaly_detection.detectors.mp_scorers import (
    FlagsAndScores,
    MPLowVarianceScorer,
    MPScorer,
)
from seer.anomaly_detection.models import AlgoConfig, AnomalyDetectionConfig
from seer.dependency_injection import inject, injected


class MPCascadingScorer(MPScorer):
    """
    This class implements a cascading scoring mechanism for Matrix Profile-based anomaly detection.
    It applies multiple scorers in sequence, returning the result of the first scorer that produces a valid output.
    This approach allows for fallback strategies and potentially more robust anomaly detection.

    The default implementation uses the LowVarianceScorer and the MPBoxCoxScorer.
    """

    scorers: list[MPScorer] = Field(
        [MPLowVarianceScorer(), MPBoxCoxScorer()], description="The list of scorers to cascade"
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
            )
            if flags_and_scores is not None:
                return flags_and_scores
        return None
