import abc

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class MPScorer(BaseModel, abc.ABC):
    @abc.abstractmethod
    def score(
        self, ts: npt.NDArray, mp: npt.NDArray, mp_dist: npt.NDArray, window_size: int
    ) -> tuple:
        return NotImplemented


class MPIRQScorer(MPScorer):
    def score(
        self, ts: npt.NDArray, mp: npt.NDArray, mp_dist: npt.NDArray, window_size: int
    ) -> tuple:
        def to_flag(score, threshold_lower, threshold_upper):
            if np.isnan(score):
                return "none"
            if score < threshold_lower:
                return "none"
            if score < threshold_upper:
                return "anomaly_low"
            return "anomaly_high"

        mp_dist_finite = mp_dist[np.isfinite(mp_dist)]

        [Q1, Q3] = np.quantile(mp_dist_finite, [0.25, 0.75])
        IQR_L = Q3 - Q1
        threshold_lower = Q3 + (1.5 * IQR_L)

        [Q1, Q3] = np.quantile(mp_dist_finite, [0.15, 0.85])
        IQR_L = Q3 - Q1
        threshold_upper = Q3 + (1.5 * IQR_L)

        scores = [
            np.NAN if np.isnan(val) or np.isinf(val) else val - threshold_upper for val in mp_dist
        ]
        flags = [to_flag(score, threshold_lower, threshold_upper) for score in scores]
        return scores, flags
