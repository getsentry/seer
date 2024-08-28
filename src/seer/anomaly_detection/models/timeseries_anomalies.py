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
    def get_anomaly_algo_data(self) -> List[Optional[Dict]]:
        return NotImplemented

    # @root_validator(pre=False)
    # def validate_all_fields_at_the_same_time(cls, field_values):
    #     if len(field_values.get("types")) != len(field_values.get("scores")):
    #         raise ValidationError("Scores and types need to be of the same length.")

    #     return field_values  # this is the value written to the class field


class MPTimeSeriesAnomalies(TimeSeriesAnomalies):
    flags: list[str] = Field(..., description="Anomaly flags")

    scores: list[float] = Field(..., description="Anomaly scores")

    matrix_profile: npt.NDArray = Field(
        ..., description="The matrix profile of the time series using which anomalies were detected"
    )

    window_size: int = Field(..., description="Window size used to build the matrix profile")

    def get_anomaly_algo_data(self) -> List[Optional[Dict]]:
        # Matrix profile is not available for the first (window_size -1) time steps. Need to set them to None.
        algo_data: List[Optional[Dict]] = [None] * (self.window_size - 1)

        for dist, index, l_index, _ in self.matrix_profile:
            algo_data.append({"mp_dist": dist, "mp_index": index, "mp_left_index": l_index})
        return algo_data
