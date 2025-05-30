import datetime
import logging

from pydantic import BaseModel

from seer.anomaly_detection.models.cleanup_predict import CleanupPredictConfig
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.anomaly_detection.models.timeseries import ProphetPrediction, TimeSeries
from seer.anomaly_detection.models.timeseries_anomalies import TimeSeriesAnomalies
from seer.db import TaskStatus

logger = logging.getLogger(__name__)


class DynamicAlert(BaseModel):
    organization_id: int
    project_id: int
    external_alert_id: int
    external_alert_source_id: int | None = None
    external_alert_source_type: int | None = None
    config: AnomalyDetectionConfig
    timeseries: TimeSeries
    anomalies: TimeSeriesAnomalies
    prophet_predictions: ProphetPrediction
    cleanup_predict_config: CleanupPredictConfig
    only_suss: bool
    data_purge_flag: TaskStatus
    last_queued_at: datetime.datetime | None
