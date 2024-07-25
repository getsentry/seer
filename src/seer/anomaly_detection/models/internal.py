import abc
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from seer.db import DbDynamicAlert

AnomalyFlags = Literal["none", "anomaly_low", "anomaly_high", "no_data"]
Sensitivities = Literal["low", "medium", "high"]
TimePeriods = Literal[15, 30, 60]
Directions = Literal["up", "down", "both"]
Seasonalities = Literal["hourly", "daily", "weekly", "auto"]


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


class TimeSeries(BaseModel):
    timestamps: npt.NDArray[np.float64] = Field(
        ...,
        description="Time stamps for the timeseries. There should be one-to-one correspondence between the timestamps and the values.",
    )
    values: npt.NDArray[np.float64] = Field(
        ...,
        description="The timeseries values. There should be one-to-one correspondence between the timestamps and the values.",
    )

    anomalies: Optional[TimeSeriesAnomalies] = Field(
        None,
        description="Anomalies identified in time series",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    # @root_validator(pre=False)
    # def validate_lengths_should_match(cls, field_values):
    #     # Do the validation instead of printing
    #     if len(field_values.get("timestamps")) != len(field_values.get("values")):
    #         raise ValidationError("TIme stamps and the values need to be of the same length.")

    #     return field_values  # this is the value written to the class field


class MPTimeSeries(TimeSeries):
    timestamps: npt.NDArray[np.float64] = Field(
        ...,
        description="Time stamps for the timeseries. There should be one-to-one correspondence between the timestamps and the values.",
    )
    values: npt.NDArray[np.float64] = Field(
        ...,
        description="The timeseries values. There should be one-to-one correspondence between the timestamps and the values.",
    )

    anomalies: Optional[MPTimeSeriesAnomalies] = Field(
        None,
        description="Anomalies identified in time series",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class DynamicAlert(BaseModel):
    id: int
    organization_id: int
    project_id: int
    external_alert_id: int
    config: dict

    @classmethod
    def from_db(cls, db_repo: DbDynamicAlert) -> "DynamicAlert":
        return cls(
            id=db_repo.id,
            organization_id=db_repo.organization_id,
            project_id=db_repo.project_id,
            external_alert_id=db_repo.external_alert_id,
            config=db_repo.config,
        )

    def to_db_model(self) -> DbDynamicAlert:
        return DbDynamicAlert(
            id=self.id,
            organization_id=self.organization_id,
            project_id=self.project_id,
            external_alert_id=self.external_alert_id,
            config=self.config,
        )
