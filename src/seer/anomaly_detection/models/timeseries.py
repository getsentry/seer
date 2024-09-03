from typing import Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.models.timeseries_anomalies import (
    MPTimeSeriesAnomalies,
    TimeSeriesAnomalies,
)


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

    def get_anomaly_algo_data(self) -> Optional[dict]:
        return None

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

    def get_anomaly_algo_data(self) -> Optional[dict]:
        return None if self.anomalies is None else {"window_size": self.anomalies.window_size}
