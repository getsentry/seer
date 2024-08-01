import abc
import logging

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MPScorer(BaseModel, abc.ABC):
    """
    Abstract base class for calculating an anomaly score
    """

    @abc.abstractmethod
    def score(
        self, mp_dist_to_score: npt.NDArray, mp_dist_baseline: npt.NDArray | None = None
    ) -> tuple:
        return NotImplemented


class MPIRQScorer(MPScorer):
    def score(
        self, mp_dist_to_score: npt.NDArray, mp_dist_baseline: npt.NDArray | None = None
    ) -> tuple:
        """
        Scores anomalies by computing the distance of the relevant MP distance from quartiles. This approach is not swayed by
        extreme values in MP distances. It also converts the score to a flag with a more meaningful interpretation of score.

        The interquartile ranges for scoring are computed using the distances passed in as mp_dist_baseline. If mp_dist_baseline is
        None then the interquartile ranges are computed from mp_dist_to_score

        Parameters:
        mp_dist_to_score: Numpy array
            The matrix profile distances that need scoring

        mp_dist_baseline: Numpy array
            Baseline distances used for calculating inter quartile range

        Returns:
            tuple with list of scores and list of flags, where each flag is one of
            * "none" - indicating not an anomaly
            * "anomaly_lower_confidence" - indicating anomaly but only with a lower threshold
            * "anomaly_higher_confidence" - indicating anomaly with a higher threshold
        """
        # Stumpy returns inf for the first timeseries[0:window_size - 2] entries. We just need to ignore those before scoring.
        if mp_dist_baseline is None:
            mp_dist_baseline_finite = mp_dist_to_score[np.isfinite(mp_dist_to_score)]
        else:
            mp_dist_baseline_finite = mp_dist_baseline[np.isfinite(mp_dist_baseline)]

        # Compute the quantiles for two different threshold levels
        [Q1, Q3] = np.quantile(mp_dist_baseline_finite, [0.25, 0.75])
        IQR_L = Q3 - Q1
        threshold_lower = Q3 + (1.5 * IQR_L)

        [Q1, Q3] = np.quantile(mp_dist_baseline_finite, [0.15, 0.85])
        IQR_L = Q3 - Q1
        threshold_upper = Q3 + (1.5 * IQR_L)

        # Compute score and anomaly flags
        scores = []
        flags = []
        for val in mp_dist_to_score:
            scores.append(0.0 if np.isnan(val) or np.isinf(val) else val - threshold_upper)
            flags.append(self._to_flag(val, threshold_lower, threshold_upper))

        return scores, flags

    def _to_flag(self, mp_dist, threshold_lower, threshold_upper):
        if np.isnan(mp_dist):
            return "none"
        if mp_dist < threshold_lower:
            return "none"
        if mp_dist < threshold_upper:
            return "anomaly_lower_confidence"
        return "anomaly_higher_confidence"
