import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TimeSeriesPoint(BaseModel):
    timestamp: float
    value: float
    anomaly_score: Optional[float] = None


class ADConfig(BaseModel):
    time_period: int = Field(
        ...,
        description="Aggregation window used in the time period, in minutes",
    )
    display_window: int = Field(..., description="Window for the view, in minutes")
    detection_threshold: Literal["low", "medium", "high"] = Field(
        ...,
        description="Low means more anomalies will be detected while high means less anomalies will be detected.",
    )
    direction: Literal["up", "down", "both"] = Field(
        ...,
        description="Identifies the type of deviation(s) to detect. Up means only anomalous values above normal values are identified while down means values lower than normal values are identified. Passing both will identify both above and below normal values.",
    )
    expected_cyclicality: Literal["hourly", "daily", "weekly", "auto"] = Field(
        ...,
        description="Underlying cyclicality in the time series. Auto means the system will detect by itself.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Alert(BaseModel):
    id: int
    cur_window: Optional[TimeSeriesPoint] = Field(
        None, description="Timestamp and the measured value for current time window."
    )


class AlertAnomaliesRequest(BaseModel):
    organization_id: int
    project_id: int
    config: ADConfig
    alert: Alert


class AlertAnomaliesResponse(BaseModel):
    anomalies: List[TimeSeriesPoint]


class StoreDataRequest(BaseModel):
    alert_id: int
    organization_id: int
    project_id: int
    timeseries: List[TimeSeriesPoint]


logger = logging.getLogger("anomaly_detection")


class AnomalyDetection:
    def __init__(self):
        pass

    def detect_anomalies(self, request: AlertAnomaliesRequest) -> AlertAnomaliesResponse:
        logger.info(f"Detecting anomalies for alert ID: {request.alert.id}")
        # Placeholder for actual anomaly detection logic
        anomalies = [
            TimeSeriesPoint(timestamp=point[0], value=point[1], anomaly_score=0.5)
            for point in request.alert.cur_window or []
        ]
        return AlertAnomaliesResponse(anomalies=anomalies)

    def store_data(self, request: StoreDataRequest) -> bool:
        logger.info(f"Storing data for request: {request}")
        return True
