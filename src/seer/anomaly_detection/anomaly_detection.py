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
from seer.anomaly_detection.models import (
    Alert,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
    TimeSeriesPoint,
)

logger = logging.getLogger("anomaly_detection")


class AnomalyDetection(BaseModel):
    def detect_anomalies(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        if isinstance(request.context, Alert):
            logger.info(f"Detecting anomalies for alert ID: {request.context.id}")
            ts = (
                [
                    TimeSeriesPoint(
                        timestamp=request.context.cur_window.timestamp,
                        value=request.context.cur_window.value,
                    )
                ]
                if request.context.cur_window
                else []
            )
            dummy_detector = DummyAnomalyDetector()
            updated = dummy_detector.detect(ts)
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
            updated = batch_detector.detect(ts)

        return DetectAnomaliesResponse(timeseries=updated)

    def store_data(self, request: StoreDataRequest) -> StoreDataResponse:
        logger.info(f"Storing data for request: {request}")
        return StoreDataResponse(success=True)
