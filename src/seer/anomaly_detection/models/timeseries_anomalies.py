import abc
import logging
from typing import Dict, List, Optional

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TimeSeriesAnomalies(BaseModel, abc.ABC):
    """
    Abstract base class for storing anomaly flags and scores. Need to extend this class to store additional information needed for
    each anomaly detection algorithm.
    """

    flags: list[str] = Field(..., description="Anomaly flags")

    scores: list[float] = Field(..., description="Anomaly scores")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @abc.abstractmethod
    def get_anomaly_algo_data(self, front_pad_to_len: int) -> List[Optional[Dict]]:
        return NotImplemented

    # @root_validator(pre=False)
    # def validate_all_fields_at_the_same_time(cls, field_values):
    #     if len(field_values.get("types")) != len(field_values.get("scores")):
    #         raise ValidationError("Scores and types need to be of the same length.")

    #     return field_values  # this is the value written to the class field


class MPTimeSeriesAnomalies(TimeSeriesAnomalies):
    flags: list[str] = Field(..., description="Anomaly flags")

    scores: list[float] = Field(..., description="Anomaly scores")

    thresholds: list[float] = Field(..., description="Score thresholds")

    matrix_profile: npt.NDArray = Field(
        ..., description="The matrix profile of the time series using which anomalies were detected"
    )

    window_size: int = Field(..., description="Window size used to build the matrix profile")

    original_flags: list[str | None] = Field(
        default=[], description="The original flags of the time series"
    )

    def get_anomaly_algo_data(self, front_pad_to_len: int) -> List[Optional[Dict]]:
        algo_data: List[Optional[Dict]] = []
        if len(self.matrix_profile) < front_pad_to_len:
            algo_data = [None] * (front_pad_to_len - len(self.matrix_profile))

        for i, (dist, index, l_index, r_index) in enumerate(self.matrix_profile):
            original_flag = self.original_flags[i] if i < len(self.original_flags) else "none"
            algo_data.append(
                {
                    "dist": dist,
                    "idx": index,
                    "l_idx": l_index,
                    "r_idx": r_index,
                    "original_flag": original_flag,
                }
            )

        return algo_data

    @staticmethod
    def extract_algo_data(map: dict):
        return (
            map.get("dist"),
            map.get("idx"),
            map.get("l_idx"),
            map.get("r_idx"),
            map.get("original_flag"),
        )
