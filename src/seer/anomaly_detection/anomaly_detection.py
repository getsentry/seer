import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Anomaly(BaseModel):
    anomaly_type: Literal["none", "anomaly_low", "anomaly_high", "no_data"] = Field(
        ...,
        description="Indicates result of the anomaly detection algorithm. 'none' means no anomaly detected, 'anomaly_low' means lower threshold, 'anomaly_high' means higher threshold, 'no_data' means time series did not have enough data to run anomaly detection.",
    )

    anomaly_score: Optional[float] = Field(None, description="Computed anomaly score")


class TimeSeriesPoint(BaseModel):
    timestamp: float
    value: float
    anomaly: Optional[Anomaly] = None


class ADConfig(BaseModel):
    time_period: Literal[15, 30, 60] = Field(
        ...,
        description="Aggregation window used in the time period, in minutes",
    )
    sensitivity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Low means more anomalies will be detected while high means less anomalies will be detected.",
    )
    direction: Literal["up", "down", "both"] = Field(
        ...,
        description="Identifies the type of deviation(s) to detect. Up means only anomalous values above normal values are identified while down means values lower than normal values are identified. Passing both will identify both above and below normal values.",
    )
    expected_seasonality: Literal["hourly", "daily", "weekly", "auto"] = Field(
        ...,
        description="Underlying cyclicality in the time series. Auto means the system will detect by itself.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Alert(BaseModel):
    id: int
    cur_window: Optional[TimeSeriesPoint] = Field(
        None, description="Timestamp and the measured value for current time window."
    )


class DetectAnomaliesRequest(BaseModel):
    organization_id: int
    project_id: int
    config: ADConfig
    context: Alert | List[TimeSeriesPoint] = Field(
        ...,
        description="Context can be an alert identified by its id or a raw time series. If alert is provided then the system will pull the related timeseries from store else it will use the provided timeseries.",
    )


class DetectAnomaliesResponse(BaseModel):
    timeseries: List[TimeSeriesPoint]


class StoreDataRequest(BaseModel):
    organization_id: int
    project_id: int
    alert: Alert
    config: ADConfig
    timeseries: List[TimeSeriesPoint]


class StoreDataResponse(BaseModel):
    success: bool


logger = logging.getLogger("anomaly_detection")


class AnomalyDetection:
    def __init__(self):
        pass

    def detect_anomalies(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        print(request)
        if isinstance(request.context, Alert):
            logger.info(f"Detecting anomalies for alert ID: {request.context.id}")
            anomalies = (
                [
                    TimeSeriesPoint(
                        timestamp=request.context.cur_window.timestamp,
                        value=request.context.cur_window.value,
                        anomaly=Anomaly(anomaly_type="none", anomaly_score=0.5),
                    )
                ]
                if request.context.cur_window
                else []
            )
        else:
            logger.info(
                f"Detecting anomalies for time series with {len(request.context)} datapoints"
            )
            anomalies = [
                TimeSeriesPoint(
                    timestamp=point.timestamp,
                    value=point.value,
                    anomaly=Anomaly(anomaly_type="none", anomaly_score=0.5),
                )
                for point in request.context or []
            ]
        # Placeholder for actual anomaly detection logic
        return DetectAnomaliesResponse(timeseries=anomalies)

    def store_data(self, request: StoreDataRequest) -> StoreDataResponse:
        logger.info(f"Storing data for request: {request}")
        return StoreDataResponse(success=True)
