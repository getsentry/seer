import logging

import numpy as np
import numpy.typing as npt
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel, Field

from seer.anomaly_detection.detectors.mp_scorers import MPScorer
from seer.anomaly_detection.detectors.normalizers import Normalizer
from seer.anomaly_detection.detectors.window_size_selectors import WindowSizeSelector
from seer.anomaly_detection.models import Anomaly, TimeSeriesPoint

logger = logging.getLogger("anomaly_detection")


class MPConfig(BaseModel):
    ignore_trivial: bool = Field(
        ...,
        description="Flag that tells the stumpy library to ignore trivial matches to speed up MP computation",
    )
    normalize_mp: bool = Field(
        ...,
        description="Flag to control of the matrix profile is normalized first",
    )


class AnomalyDetector(BaseModel):
    def detect(self, timeseries: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        raise NotImplementedError("Subclasses should implement this!")


class MPBatchAnomalyDetector(AnomalyDetector):
    config: MPConfig
    scorer: MPScorer
    ws_selector: WindowSizeSelector
    normalizer: Normalizer

    def detect(self, timeseries: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        mp, mp_dist, scores, flags, window_size = self._compute_matrix_profile(timeseries)

        self._update_ts(timeseries, scores, flags)
        return timeseries

    def _update_ts(
        self, timeseries: list[TimeSeriesPoint], scores: npt.NDArray, flags: list
    ) -> list[TimeSeriesPoint]:
        for point, score, flag in zip(timeseries, scores, flags):
            score = 0.0 if np.isnan(score) or score < 0 else score
            point.anomaly = Anomaly(anomaly_score=score, anomaly_type=flag)
        return timeseries

    def _to_raw_ts(self, timeseries: list[TimeSeriesPoint]) -> npt.NDArray:
        return np.array([np.float64(point.value) for point in timeseries])

    def _get_mp_dist_from_mp(
        self,
        mp: npt.NDArray,
        ts: npt.NDArray[np.float64],
        normalize_mp: bool,
        extrapolate_to_ts_len: bool,
    ):
        mp_dist = mp[:, 0]
        if normalize_mp:
            mp_dist = self.normalizer.normalize(mp_dist)

        if extrapolate_to_ts_len:
            nan_value_count = np.empty(len(ts) - len(mp_dist))
            nan_value_count.fill(np.nan)
            mp_dist_updated = np.concatenate((nan_value_count, mp_dist))
            return mp_dist_updated.astype(np.float64)
        else:
            return mp_dist.astype(float)

    def _compute_matrix_profile(self, timeseries: list[TimeSeriesPoint]) -> tuple:
        ts_values = self._to_raw_ts(timeseries)

        window_size = self.ws_selector.optimal_window_size(ts_values)
        logger.debug(f"window_size: {window_size}")
        if window_size <= 0:
            return None, None, window_size
        # Get the matrix profile for the time series
        mp = stumpy.stump(
            ts_values,
            m=max(3, window_size),
            ignore_trivial=self.config.ignore_trivial,
            normalize=False,
        )

        mp_dist = self._get_mp_dist_from_mp(
            mp, ts_values, self.config.normalize_mp, extrapolate_to_ts_len=True
        )

        scores, flags = self.scorer.score(ts_values, mp, mp_dist, window_size)
        return mp, mp_dist, scores, flags, window_size


class DummyAnomalyDetector(AnomalyDetector):

    def detect(self, timeseries: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        return [
            TimeSeriesPoint(
                timestamp=point.timestamp,
                value=point.value,
                anomaly=Anomaly(anomaly_type="none", anomaly_score=0.5),
            )
            for point in timeseries or []
        ]
