import abc

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class MPScorer(BaseModel, abc.ABC):
    """
    Abstract base class for calculating an anomaly score
    """

    @abc.abstractmethod
    def score(self, mp: npt.NDArray, mp_dist: npt.NDArray, score_from: int | None) -> tuple:
        return NotImplemented


class MPIRQScorer(MPScorer):
    def score(self, mp: npt.NDArray, mp_dist: npt.NDArray, score_from: int | None) -> tuple:
        """
        Concrete implementation of MPScorer that scores anomalies by computing the distance of the relevant MP distance from the
        mean using inter quartile range. This approach is not swayed by extreme values in MP distances. It also converts the score
        to a flag with a more meaningful interpretation of score

        Parameters:
        ts: Numpy array
            The time series

        mp: Numpy array
            The matrix profile as returned by the call to stumpy.stump method

        mp_dist: Numpy array
            The matrix profile distances for the timeseries

        window_size: int
            The window size used to compute matrix profile

        Returns:
            tuple with list of scores and list of flags, where each flag is one of
            * "none" - indicating not an anomaly
            * "anomaly_low" - indicating anomaly but only with a lower threshold
            * "anomaly_high" - indicating anomaly with a higher threshold
        """

        def to_flag(mp_dist, threshold_lower, threshold_upper):
            if np.isnan(mp_dist):
                return "none"
            if mp_dist < threshold_lower:
                return "none"
            if mp_dist < threshold_upper:
                return "anomaly_low"
            return "anomaly_high"

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
        for val in mp_dist[0 if score_from is None else score_from :]:
            scores.append(0.0 if np.isnan(val) or np.isinf(val) else val - threshold_upper)
            flags.append(to_flag(val, threshold_lower, threshold_upper))

        return scores, flags
