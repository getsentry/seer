from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AnomalyFlags = Literal["none", "anomaly_lower_confidence", "anomaly_higher_confidence", "no_data"]
Sensitivities = Literal["low", "medium", "high"]
TimePeriods = Literal[15, 30, 60]
Directions = Literal["up", "down", "both"]
Seasonalities = Literal["hourly", "daily", "weekly", "auto"]


class Anomaly(BaseModel):
    anomaly_type: AnomalyFlags = Field(
        ...,
        description="Indicates result of the anomaly detection algorithm. 'none' means no anomaly detected, 'anomaly_lower_confidence' means lower threshold, 'anomaly_higher_confidence' means higher threshold, 'no_data' means time series did not have enough data to run anomaly detection.",
    )

    anomaly_score: Optional[float] = Field(None, description="Computed anomaly score")


class TimeSeriesPoint(BaseModel):
    timestamp: float
    value: float
    anomaly: Optional[Anomaly] = None


class AnomalyDetectionConfig(BaseModel):
    time_period: TimePeriods = Field(
        ...,
        description="Aggregation window used in the time period, in minutes",
    )
    sensitivity: Sensitivities = Field(
        ...,
        description="Low means more anomalies will be detected while high means less anomalies will be detected.",
    )
    direction: Directions = Field(
        ...,
        description="Identifies the type of deviation(s) to detect. Up means only anomalous values above normal values are identified while down means values lower than normal values are identified. Passing both will identify both above and below normal values.",
    )
    expected_seasonality: Seasonalities = Field(
        ...,
        description="Underlying cyclicality in the time series. Auto means the system will detect by itself.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AlertInSeer(BaseModel):
    id: int
    cur_window: Optional[TimeSeriesPoint] = Field(
        None, description="Timestamp and the measured value for current time window."
    )


class DetectAnomaliesRequest(BaseModel):
    organization_id: int
    project_id: int
    config: AnomalyDetectionConfig
    context: AlertInSeer | List[TimeSeriesPoint] = Field(
        ...,
        description="Context can be an alert identified by its id or a raw time series. If alert is provided then the system will pull the related timeseries from store else it will use the provided timeseries.",
    )


class DetectAnomaliesResponse(BaseModel):
    timeseries: List[TimeSeriesPoint]


class StoreDataRequest(BaseModel):
    organization_id: int
    project_id: int
    alert: AlertInSeer
    config: AnomalyDetectionConfig
    timeseries: List[TimeSeriesPoint]


class StoreDataResponse(BaseModel):
    success: bool
