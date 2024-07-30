import abc

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


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
