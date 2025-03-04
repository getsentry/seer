from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
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
    y: npt.NDArray[np.float64] = Field(
        ...,
        description="The actual values. There should be one-to-one correspondence between the timestamps and the values.",
    )
    yhat: npt.NDArray[np.float64] = Field(
        ...,
        description="The predicted values. There should be one-to-one correspondence between the timestamps and the values.",
    )
    yhat_lower: npt.NDArray[np.float64] = Field(
        ...,
        description="The lower bound of the predicted values. There should be one-to-one correspondence between the timestamps and the values.",
    )
    yhat_upper: npt.NDArray[np.float64] = Field(
        ...,
        description="The upper bound of the predicted values. There should be one-to-one correspondence between the timestamps and the values.",
    )

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "ds": pd.Series(self.timestamps, dtype=np.float64),
                "y": pd.Series(self.y, dtype=np.float64),
                "actual": pd.Series(self.y, dtype=np.float64),
                "yhat": pd.Series(self.yhat, dtype=np.float64),
                "yhat_lower": pd.Series(self.yhat_lower, dtype=np.float64),
                "yhat_upper": pd.Series(self.yhat_upper, dtype=np.float64),
            },
        )
        df.sort_values(by="ds", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def from_prophet_df(prophet_df: pd.DataFrame) -> "ProphetPrediction":

        return ProphetPrediction(
            timestamps=np.array([pd.to_datetime(date, unit="s") for date in prophet_df.ds]),
            y=np.array(prophet_df.y),
            yhat=np.array(prophet_df.yhat),
            yhat_lower=np.array(prophet_df.yhat_lower),
            yhat_upper=np.array(prophet_df.yhat_upper),
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
