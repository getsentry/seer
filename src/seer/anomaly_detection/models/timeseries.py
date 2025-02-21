from typing import Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.models.timeseries_anomalies import (
    MPTimeSeriesAnomalies,
    TimeSeriesAnomalies,
)


class ProphetPrediction(BaseModel):
    timestamps: npt.NDArray[np.float64] = Field(
        ...,
        description="Time stamps for the timeseries. There should be one-to-one correspondence between the timestamps and the values.",
    )
    yhat: npt.NDArray[np.float64] = Field(
        ...,
        description="The timeseries values. There should be one-to-one correspondence between the timestamps and the values.",
    )
    yhat_lower: npt.NDArray[np.float64] = Field(
        ...,
        description="The timeseries values. There should be one-to-one correspondence between the timestamps and the values.",
    )
    yhat_upper: npt.NDArray[np.float64] = Field(
        ...,
        description="The timeseries values. There should be one-to-one correspondence between the timestamps and the values.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
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

    prophet_predictions: Optional[ProphetPrediction] = Field(
        None,
        description="Prophet prediction for the timeseries",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_anomaly_algo_data(self) -> Optional[dict]:
        return None


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
