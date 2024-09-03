import abc
import logging

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from seer.anomaly_detection.models import AnomalyFlags, Directions, Sensitivities

logger = logging.getLogger(__name__)


class MPScorer(BaseModel, abc.ABC):
    """
    Abstract base class for calculating an anomaly score
    """

    @abc.abstractmethod
    def batch_score(
        self,
        ts: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> tuple:
        return NotImplemented

    @abc.abstractmethod
    def stream_score(
        self,
        ts_value: float,
        mp_dist: float,
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
        ts_baseline: npt.NDArray[np.float64],
        mp_dist_baseline: npt.NDArray[np.float64],
    ) -> tuple:
        return NotImplemented


class MPIRQScorer(MPScorer):

    def batch_score(
        self,
        ts: npt.NDArray[np.float64],
        mp_dist: npt.NDArray[np.float64],
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
    ) -> tuple:
        """
        Scores anomalies by computing the distance of the relevant MP distance from quartiles. This approach is not swayed by
        extreme values in MP distances. It also converts the score to a flag with a more meaningful interpretation of score.

        The interquartile ranges for scoring are computed using the distances passed in as mp_dist_baseline. If mp_dist_baseline is
        None then the interquartile ranges are computed from mp_dist_to_score

        Parameters:
        mp_dist_to_score: Numpy array
            The matrix profile distances that need scoring
        sensitivity: Sensitivities
            Low sensitivity will detect more anomalies with more false positives and high sensitiviy will detect less anomalies with more false negatives
        direction: Directions
            Up will detect anomaly only if the detected anomaly is in the upward direction while Down will detect it only if it is downward. Both will cover both up and down.

        Returns:
            tuple with list of scores and list of flags, where each flag is one of
            * "none" - indicating not an anomaly
            * "anomaly_lower_confidence" - indicating anomaly but only with a lower threshold
            * "anomaly_higher_confidence" - indicating anomaly with a higher threshold
        """
        # Stumpy returns inf for the first timeseries[0:window_size - 2] entries. We just need to ignore those before scoring.
        mp_dist_baseline_finite = mp_dist[np.isfinite(mp_dist)]

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
        for i, val in enumerate(mp_dist):
            scores.append(0.0 if np.isnan(val) or np.isinf(val) else val - threshold_upper)
            flag = self._to_flag(val, threshold_lower, threshold_upper)
            if i > 2 * window_size:
                flag = self._adjust_flag_for_vicinity(
                    flag=flag, ts_value=ts[i], context=ts[i - 2 * window_size : i - 1]
                )
            flags.append(flag)

        return scores, flags

    def stream_score(
        self,
        ts_value: float,
        mp_dist: float,
        sensitivity: Sensitivities,
        direction: Directions,
        window_size: int,
        ts_baseline: npt.NDArray[np.float64],
        mp_dist_baseline: npt.NDArray[np.float64],
    ) -> tuple:
        """
        Scores anomalies by computing the distance of the relevant MP distance from quartiles. This approach is not swayed by
        extreme values in MP distances. It also converts the score to a flag with a more meaningful interpretation of score.

        The interquartile ranges for scoring are computed using the distances passed in as mp_dist_baseline. If mp_dist_baseline is
        None then the interquartile ranges are computed from mp_dist_to_score

        Parameters:
        mp_dist_to_score: Numpy array
            The matrix profile distances that need scoring
        sensitivity: Sensitivities
            Low sensitivity will detect more anomalies with more false positives and high sensitiviy will detect less anomalies with more false negatives
        direction: Directions
            Up will detect anomaly only if the detected anomaly is in the upward direction while Down will detect it only if it is downward. Both will cover both up and down.
        mp_dist_baseline: Numpy array
            Baseline distances used for calculating inter quartile range

        Returns:
            tuple with list of scores and list of flags, where each flag is one of
            * "none" - indicating not an anomaly
            * "anomaly_lower_confidence" - indicating anomaly but only with a lower threshold
            * "anomaly_higher_confidence" - indicating anomaly with a higher threshold
        """
        # Stumpy returns inf for the first timeseries[0:window_size - 2] entries. We just need to ignore those before scoring.
        mp_dist_baseline_finite = mp_dist_baseline[np.isfinite(mp_dist_baseline)]

        # Compute the quantiles for two different threshold levels
        [Q1, Q3] = np.quantile(mp_dist_baseline_finite, [0.25, 0.75])
        IQR_L = Q3 - Q1
        threshold_lower = Q3 + (1.5 * IQR_L)

        [Q1, Q3] = np.quantile(mp_dist_baseline_finite, [0.15, 0.85])
        IQR_L = Q3 - Q1
        threshold_upper = Q3 + (1.5 * IQR_L)

        # Compute score and anomaly flags
        score = 0.0 if np.isnan(mp_dist) or np.isinf(mp_dist) else mp_dist - threshold_upper
        flag = self._to_flag(mp_dist, threshold_lower, threshold_upper)
        # anomaly identified. apply logic to check for peak and trough
        flag = self._adjust_flag_for_vicinity(flag, ts_value, ts_baseline[-2 * window_size :])
        return [score], [flag]

    def _to_flag(self, mp_dist: float, threshold_lower: float, threshold_upper: float):
        if np.isnan(mp_dist):
            return "none"
        if mp_dist < threshold_lower:
            return "none"
        if mp_dist < threshold_upper:
            return "anomaly_lower_confidence"
        return "anomaly_higher_confidence"

    def _adjust_flag_for_vicinity(
        self, ts_value: float, flag: AnomalyFlags, context: npt.NDArray[np.float64]
    ) -> AnomalyFlags:
        """
        This method adjusts the severity of a detected anomaly based on the underlying time step's proximity to peaks and troughs.
        The intuition is that for our alerting and metrics use case, an anomaly near peak or trough is more critical than one that is not.
        Current approach is to use the inter quartile range from a subsequence of the time series, identified by the context parameter.

        Parameters:
        ts_value: float
            The time step being analyzied for anomaly

        flag: AnomalyFlags
            Anomaly identified before applying this peak-trough logic

        context: npt.NDArray[np.float64]
            Time series subsequence that is used for teak-trough detection

        """
        if flag == "anomaly_higher_confidence" or flag == "anomaly_lower_confidence":
            [Q1, Q3] = np.quantile(context, [0.25, 0.75])
            IQR = Q3 - Q1
            threshold_lower = Q1 - (0 * IQR)
            threshold_upper = Q3 + (0 * IQR)
            # if ts_value > Q1 and ts_value < Q3:
            if ts_value >= threshold_lower and ts_value <= threshold_upper:
                flag = "anomaly_lower_confidence"
            else:
                flag = "anomaly_higher_confidence"
        return flag
