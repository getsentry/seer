import abc
import logging

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from seer.anomaly_detection.detectors.mp_utils import MPUtils
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class MPScorer(BaseModel, abc.ABC):
    """
    Abstract base class for calculating an anomaly score
    """

    @abc.abstractmethod
    def score(self, mp: npt.NDArray, mp_dist: npt.NDArray) -> tuple:
        return NotImplemented

    @abc.abstractmethod
    def stream_score(self, ts: npt.NDArray, mp: npt.NDArray, mp_dist: npt.NDArray) -> tuple:
        return NotImplemented


class MPIRQScorer(MPScorer):
    def score(self, mp: npt.NDArray, mp_dist: npt.NDArray) -> tuple:
        """
        Concrete implementation of MPScorer that scores anomalies by computing the distance of the relevant MP distance from the
        mean using inter quartile range. This approach is not swayed by extreme values in MP distances. It also converts the score
        to a flag with a more meaningful interpretation of score

        Parameters:
        mp: Numpy array
            The full matrix profile as returned by the call to stumpy.stump method

        mp_dist: Numpy array
            The matrix profile distances that need scoring

        Returns:
            tuple with list of scores and list of flags, where each flag is one of
            * "none" - indicating not an anomaly
            * "anomaly_lower_confidence" - indicating anomaly but only with a lower threshold
            * "anomaly_higher_confidence" - indicating anomaly with a higher threshold
        """

        if len(mp_dist[~np.isfinite(mp_dist)]) > 0:
            # TODO: Add sentry logging and metric here
            pass

        mp_dist_finite = mp_dist[np.isfinite(mp_dist)]

        [Q1, Q3] = np.quantile(mp_dist_finite, [0.25, 0.75])
        IQR_L = Q3 - Q1
        threshold_lower = Q3 + (1.5 * IQR_L)

        [Q1, Q3] = np.quantile(mp_dist_finite, [0.15, 0.85])
        IQR_L = Q3 - Q1
        threshold_upper = Q3 + (1.5 * IQR_L)
        scores = []
        flags = []
        for val in mp_dist:
            scores.append(0.0 if np.isnan(val) or np.isinf(val) else val - threshold_upper)
            flags.append(self._to_flag(val, threshold_lower, threshold_upper))

        return scores, flags

    @inject
    def stream_score(self, ts, mp, mp_dist, mp_utils: MPUtils = injected):
        mp_full_dist = mp_utils.get_mp_dist_from_mp(mp, pad_to_len=len(ts))

        # Stumpy returns inf for the first timeseries[0:window_size - 2] entries. We just need to ignore those before scoring.
        mp_full_dist_finite = mp_full_dist[np.isfinite(mp_full_dist)]

        [Q1, Q3] = np.quantile(mp_full_dist_finite, [0.25, 0.75])
        IQR_L = Q3 - Q1
        threshold_lower = Q3 + (1.5 * IQR_L)

        [Q1, Q3] = np.quantile(mp_full_dist_finite, [0.15, 0.85])
        IQR_L = Q3 - Q1
        threshold_upper = Q3 + (1.5 * IQR_L)
        scores = [
            np.nan if np.isnan(val) or np.isinf(val) else val - threshold_upper for val in mp_dist
        ]

        flags = [self._to_flag(val, threshold_lower, threshold_upper) for val in mp_dist]
        return scores, flags

    def _to_flag(self, mp_dist, threshold_lower, threshold_upper):
        if np.isnan(mp_dist):
            return "none"
        if mp_dist < threshold_lower:
            return "none"
        if mp_dist < threshold_upper:
            return "anomaly_lower_confidence"
        return "anomaly_higher_confidence"
