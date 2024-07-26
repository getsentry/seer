import logging

from pydantic import BaseModel

from seer.anomaly_detection.detectors import (
    DummyAnomalyDetector,
    MinMaxNormalizer,
    MPBatchAnomalyDetector,
    MPConfig,
    MPIRQScorer,
    SuSSWindowSizeSelector,
)
from seer.anomaly_detection.models.converters import (
    convert_external_ts_to_internal,
    store_request_to_dynamic_alert,
)
from seer.anomaly_detection.models.external import (
    AlertInSeer,
    Anomaly,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
    TimeSeriesPoint,
)

logger = logging.getLogger(__name__)


class AnomalyDetection(BaseModel):
    def detect_anomalies(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        if isinstance(request.context, AlertInSeer):
            logger.info(f"Detecting anomalies for alert ID: {request.context.id}")
            ts = []
            if request.context.cur_window:
                ts.append(
                    TimeSeriesPoint(
                        timestamp=request.context.cur_window.timestamp,
                        value=request.context.cur_window.value,
                    )
                )
            dummy_detector = DummyAnomalyDetector()
            ts_updated = dummy_detector.detect(convert_external_ts_to_internal(ts))
        else:
            logger.info(
                f"Detecting anomalies for time series with {len(request.context)} datapoints"
            )
            ts = request.context
            batch_detector = MPBatchAnomalyDetector(
                config=MPConfig(ignore_trivial=True, normalize_mp=False),
                scorer=MPIRQScorer(),
                ws_selector=SuSSWindowSizeSelector(),
                normalizer=MinMaxNormalizer(),
            )
            ts_updated = batch_detector.detect(convert_external_ts_to_internal(ts))

        for i, point in enumerate(ts):
            if ts_updated.anomalies is None:
                raise Exception("No anomalies available for the timeseries.")
            point.anomaly = Anomaly(
                anomaly_score=ts_updated.anomalies.scores[i],
                anomaly_type=ts_updated.anomalies.flags[i],
            )
        return DetectAnomaliesResponse(timeseries=ts)

    def store_data(self, request: StoreDataRequest) -> StoreDataResponse:

        logger.info(
            "store_alert_request",
            extra={
                "organization_id": request.organization_id,
                "project_id": request.project_id,
                "external_alert_id": request.alert.id,
            },
        )
        dynamic_alert = store_request_to_dynamic_alert(request)
        dynamic_alert.save(request.timeseries)
        return StoreDataResponse(success=True)
