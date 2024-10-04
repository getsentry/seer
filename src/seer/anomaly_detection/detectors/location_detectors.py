import abc
import logging
from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
import sentry_sdk
from prophet import Prophet  # type: ignore
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PointLocation(Enum):
    UP = 1
    DOWN = 2
    NONE = 3


class LocationDetector(BaseModel, abc.ABC):
    """
    Abstract base class for detecting relative location of a value in a time series.
    """

    @abc.abstractmethod
    def detect(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
    ) -> PointLocation:
        return NotImplemented


class LinearRegressionLocationDetector(LocationDetector):
    """
    Detects relative location of the streamed value in the context of recent data points using linear regression.
    """

    window_size: int = Field(
        default=10, description="Number of recent data points to consider for location detection"
    )
    threshold: float = Field(
        default=0.5,
        description="Slope threshold to determine if a location significantly deviates from the trend",
    )

    def detect(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
    ) -> PointLocation:
        """
        Detect relative location of the streamed value in the context of recent data points using linear regression.

        Args:
            streamed_value (np.float64): The current value
            history_values (npt.NDArray[np.float64]): Historical time series data

        Returns:
            PointLocation: The detected relative location of the streamed value as compared to the trend of recent data points (UP, DOWN, or NONE).
            UP: The streamed value is above the trend of recent data points.
            DOWN: The streamed value is below the trend of recent data points.
            NONE: The streamed value is within the expected range of recent data points.
        """
        recent_data = np.concatenate([history_values[-self.window_size :], [streamed_value]])

        if len(recent_data) < self.window_size + 1:
            return PointLocation.NONE  # Not enough data to determine trend

        x = np.arange(len(recent_data))
        y = recent_data

        # Perform linear regression
        slope, _ = np.polyfit(x, y, 1)

        if slope > self.threshold:
            return PointLocation.UP
        elif slope < -self.threshold:
            return PointLocation.DOWN
        else:
            return PointLocation.NONE


class ProphetLocationDetector(LocationDetector):
    """
    Detects relative location of the streamed value using Facebook's Prophet forecasting model.

    This detector uses Prophet to fit a model on historical data and make a prediction
    for the current timestamp. It then compares the streamed value against the predicted
    value (or prediction interval if uncertainty samples are used) to determine if the
    point is above, below, or within the expected range.

    """

    uncertainty_samples: bool = Field(
        default=True, description="Whether to use uncertainty samples for Prophet"
    )

    @sentry_sdk.trace
    def detect(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
    ) -> PointLocation:
        """
        Detect relative location of the streamed value in the context of recent data points using Prophet.

        Args:
            streamed_value (np.float64): The current value
            streamed_timestamp (np.float64): The current timestamp
            history_values (npt.NDArray[np.float64]): Historical time series data
            history_timestamps (npt.NDArray[np.float64]): Historical time series timestamps

        Returns:
            PointLocation: The detected relative location of the streamed value as compared to the trend or expected value (UP, DOWN, or NONE).
            UP: The streamed value is above the trend or expected value.
            DOWN: The streamed value is below the trend or expected value.
            NONE: The streamed value is within the expected range of recent data points.
        """
        # Create Prophet model and fit on historical data
        model = Prophet(mcmc_samples=0, uncertainty_samples=self.uncertainty_samples)
        ts = pd.DataFrame({"ds": history_timestamps, "y": history_values})
        model.fit(ts)

        # Create future dataframe with the current timestamp for prediction
        future = pd.DataFrame({"ds": np.append(history_timestamps, [streamed_timestamp])})
        future["ds"] = future["ds"].astype(float)

        # Predict and compare with streamed value
        forecast = model.predict(future)
        if self.uncertainty_samples:
            prophet_trend_upper = forecast.loc[len(forecast) - 1]["trend_upper"]
            prophet_trend_lower = forecast.loc[len(forecast) - 1]["trend_lower"]
            if streamed_value > prophet_trend_upper:
                return PointLocation.UP
            elif streamed_value < prophet_trend_lower:
                return PointLocation.DOWN
            else:
                return PointLocation.NONE
        else:
            forecast = forecast.yhat.loc[len(forecast) - 1]
            if np.isclose(streamed_value, forecast, rtol=1e-5, atol=1e-8):
                return PointLocation.NONE
            elif streamed_value > forecast:
                return PointLocation.UP
            else:
                return PointLocation.DOWN
