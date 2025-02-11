import abc
import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import sentry_sdk
from prophet import Prophet  # type: ignore
from pydantic import BaseModel, Field

from seer.anomaly_detection.models import (
    AlgoConfig,
    PointLocation,
    RelativeLocation,
    Threshold,
    ThresholdType,
)
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


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
    ) -> Optional[RelativeLocation]:
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
    ) -> Optional[RelativeLocation]:
        """
        Detect relative location of the streamed value in the context of recent data points using linear regression.

        Args:
            streamed_value (np.float64): The current value
            history_values (npt.NDArray[np.float64]): Historical time series data

        Returns:
            RelativeLocation: The detected relative location of the streamed value as compared to the trend of recent data points (UP, DOWN, or NONE).
            UP: The streamed value is above the trend of recent data points.
            DOWN: The streamed value is below the trend of recent data points.
            NONE: The streamed value is within the expected range of recent data points.
        """
        recent_data = np.concatenate([history_values[-self.window_size :], [streamed_value]])

        if len(recent_data) < self.window_size + 1:
            return None  # Not enough data to determine trend

        x = np.arange(len(recent_data))
        y = recent_data

        # Perform linear regression
        slope, _ = np.polyfit(x, y, 1)

        if slope > self.threshold:
            return RelativeLocation(
                location=PointLocation.UP,
                thresholds=[],
            )
        elif slope < -self.threshold:
            return RelativeLocation(
                location=PointLocation.DOWN,
                thresholds=[],
            )
        else:
            return RelativeLocation(
                location=PointLocation.NONE,
                thresholds=[],
            )


class ProphetLocationDetector(LocationDetector):
    """
    Detects relative location of the streamed value using Facebook's Prophet forecasting model.

    This detector uses Prophet to fit a model on historical data and make a prediction
    for the current timestamp. It then compares the streamed value against the predicted
    value (or prediction interval if uncertainty samples are used) to determine if the
    point is above, below, or within the expected range.

    """

    @sentry_sdk.trace
    @inject
    def detect(
        self,
        streamed_value: np.float64,
        streamed_timestamp: np.float64,
        history_values: npt.NDArray[np.float64],
        history_timestamps: npt.NDArray[np.float64],
        algo_config: AlgoConfig = injected,
    ) -> Optional[RelativeLocation]:
        """
        Detect relative location of the streamed value in the context of recent data points using Prophet.

        Args:
            streamed_value (np.float64): The current value
            streamed_timestamp (np.float64): The current timestamp
            history_values (npt.NDArray[np.float64]): Historical time series data
            history_timestamps (npt.NDArray[np.float64]): Historical time series timestamps

        Returns:
            RelativeLocation: The detected relative location of the streamed value as compared to the predicted value (UP, DOWN, or NONE).
            UP: The streamed value is above the predicted value.
            DOWN: The streamed value is below the predicted value.
            NONE: The streamed value is within the expected range of recent data points.
        """
        # Create Prophet model and fit on historical data
        model = Prophet(
            mcmc_samples=algo_config.prophet_mcmc_samples,
            uncertainty_samples=algo_config.prophet_uncertainty_samples,
        )
        ts = pd.DataFrame({"ds": history_timestamps, "y": history_values})
        model.fit(ts)

        # Create future dataframe with the current timestamp for prediction
        future = pd.DataFrame({"ds": np.append(history_timestamps, [streamed_timestamp])})
        future["ds"] = future["ds"].astype(float)

        # Predict and compare with streamed value
        forecast = model.predict(future)
        if algo_config.prophet_uncertainty_samples > 0 or algo_config.prophet_mcmc_samples > 0:
            streamed_forecast = forecast.loc[len(forecast) - 1]
            yhat_upper = streamed_forecast["yhat_upper"]
            yhat_lower = streamed_forecast["yhat_lower"]
            if algo_config.return_thresholds:
                thresholds = [
                    Threshold(
                        type=ThresholdType.PREDICTION,
                        timestamp=streamed_timestamp,
                        upper=yhat_upper,
                        lower=yhat_lower,
                    ),
                    Threshold(
                        type=ThresholdType.TREND,
                        timestamp=streamed_timestamp,
                        upper=streamed_forecast["trend_upper"],
                        lower=streamed_forecast["trend_lower"],
                    ),
                ]
            else:
                thresholds = []

            if streamed_value > yhat_upper:
                return RelativeLocation(
                    location=PointLocation.UP,
                    thresholds=thresholds,
                )
            elif streamed_value < yhat_lower:
                return RelativeLocation(
                    location=PointLocation.DOWN,
                    thresholds=thresholds,
                )
            else:
                return RelativeLocation(
                    location=PointLocation.NONE,
                    thresholds=thresholds,
                )
        else:
            forecast = forecast.yhat.loc[len(forecast) - 1]
            if np.isclose(streamed_value, forecast, rtol=1e-5, atol=1e-8):
                return RelativeLocation(
                    location=PointLocation.NONE,
                    thresholds=[],
                )
            elif streamed_value > forecast:
                return RelativeLocation(
                    location=PointLocation.UP,
                    thresholds=[],
                )
            else:
                return RelativeLocation(
                    location=PointLocation.DOWN,
                    thresholds=[],
                )
