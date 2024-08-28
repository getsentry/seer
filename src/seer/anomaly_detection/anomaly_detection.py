import logging
from typing import List

import sentry_sdk
from pydantic import BaseModel

from seer.anomaly_detection.accessors import AlertDataAccessor
from seer.anomaly_detection.anomaly_detection_di import anomaly_detection_module
from seer.anomaly_detection.detectors import (
    AnomalyDetector,
    MPBatchAnomalyDetector,
    MPStreamAnomalyDetector,
)
from seer.anomaly_detection.models import TimeSeriesAnomalies
from seer.anomaly_detection.models.converters import convert_external_ts_to_internal
from seer.anomaly_detection.models.external import (
    AlertInSeer,
    Anomaly,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
    TimeSeriesPoint,
)
from seer.dependency_injection import inject, injected

anomaly_detection_module.enable()
logger = logging.getLogger(__name__)


class AnomalyDetection(BaseModel):

    @sentry_sdk.trace
    def _batch_detect(self, timeseries: List[TimeSeriesPoint]):
        logger.info(f"Detecting anomalies for time series with {len(timeseries)} datapoints")
        batch_detector: AnomalyDetector = MPBatchAnomalyDetector()
        anomalies = batch_detector.detect(convert_external_ts_to_internal(timeseries))
        self._update_anomalies(timeseries, anomalies)
        return timeseries

    @inject
    @sentry_sdk.trace
    def _online_detect(
        self,
        alert: AlertInSeer,
        alert_data_accessor: AlertDataAccessor = injected,
    ) -> List[TimeSeriesPoint]:
        logger.info(f"Detecting anomalies for alert ID: {alert.id}")
        ts_external: List[TimeSeriesPoint] = []
        if alert.cur_window:
            ts_external.append(
                TimeSeriesPoint(
                    timestamp=alert.cur_window.timestamp,
                    value=alert.cur_window.value,
                )
            )

        # Retrieve historic data
        historic = alert_data_accessor.query(alert.id)
        if historic is None:
            raise Exception(f"Invalid alert id {alert.id}")

        # TODO: Need to check the time gap between historic data and the new datapoint against the alert configuration

        # Run batch detect on history data
        # TODO: This step can be optimized further by caching the matrix profile in the database
        batch_detector = MPBatchAnomalyDetector()
        anomalies = batch_detector.detect(historic.timeseries)

        # Run stream detection
        stream_detector: AnomalyDetector = MPStreamAnomalyDetector(
            base_timestamps=historic.timeseries.timestamps,
            base_values=historic.timeseries.values,
            base_mp=anomalies.matrix_profile,
            window_size=anomalies.window_size,
        )
        streamed_anomalies = stream_detector.detect(convert_external_ts_to_internal(ts_external))
        self._update_anomalies(ts_external, streamed_anomalies)

        # Save new data point
        alert_data_accessor.save_timepoint(external_alert_id=alert.id, timepoint=ts_external[0])
        # TODO: Clean up old data
        return ts_external

    def _update_anomalies(self, ts_external: List[TimeSeriesPoint], anomalies: TimeSeriesAnomalies):
        if anomalies is None:
            raise Exception("No anomalies available for the timeseries.")
        for i, point in enumerate(ts_external):
            point.anomaly = Anomaly(
                anomaly_score=anomalies.scores[i],
                anomaly_type=anomalies.flags[i],
            )

    def detect_anomalies(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        ts: List[TimeSeriesPoint] = (
            self._online_detect(request.context)
            if isinstance(request.context, AlertInSeer)
            else self._batch_detect(request.context)
        )
        return DetectAnomaliesResponse(timeseries=ts)

    @inject
    def store_data(
        self, request: StoreDataRequest, alert_data_accessor: AlertDataAccessor = injected
    ) -> StoreDataResponse:
        logger.info(
            "store_alert_request",
            extra={
                "organization_id": request.organization_id,
                "project_id": request.project_id,
                "external_alert_id": request.alert.id,
            },
        )
        alert_data_accessor.save_alert(
            organization_id=request.organization_id,
            project_id=request.project_id,
            external_alert_id=request.alert.id,
            config=request.config,
            timeseries=request.timeseries,
        )
        return StoreDataResponse(success=True)
