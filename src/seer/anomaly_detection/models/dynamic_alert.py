import logging

from pydantic import BaseModel

from seer.anomaly_detection.models.cleanup_predict import CleanupPredictConfig
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.anomaly_detection.models.timeseries import TimeSeries
from seer.anomaly_detection.models.timeseries_anomalies import TimeSeriesAnomalies

logger = logging.getLogger(__name__)


class DynamicAlert(BaseModel):
    organization_id: int
    project_id: int
    external_alert_id: int
    config: AnomalyDetectionConfig
    timeseries: TimeSeries
    anomalies: TimeSeriesAnomalies
    cleanup_predict_config: CleanupPredictConfig
    prophet_predictions: ProphetPrediction
    only_suss: bool
