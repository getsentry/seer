import logging
from typing import List

from pydantic import BaseModel, Field

from seer.anomaly_detection.accessors import AlertDataAccessor, DbAlertDataAccessor
from seer.anomaly_detection.detectors import (
    AnomalyDetector,
    MinMaxNormalizer,
    MPBatchAnomalyDetector,
    MPConfig,
    MPIRQScorer,
    MPStreamAnomalyDetector,
    SuSSWindowSizeSelector,
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

logger = logging.getLogger(__name__)


class AnomalyDetection(BaseModel):
    alert_data_accessor: AlertDataAccessor = Field(
        DbAlertDataAccessor(),
        description="Alert data accessor for saving and retrieving alert history data",
    )

    def _batch_detect(self, timeseries: List[TimeSeriesPoint]):
        logger.info(f"Detecting anomalies for time series with {len(timeseries)} datapoints")
        batch_detector: AnomalyDetector = MPBatchAnomalyDetector(
            config=MPConfig(ignore_trivial=True, normalize_mp=False),
            scorer=MPIRQScorer(),
            ws_selector=SuSSWindowSizeSelector(),
            normalizer=MinMaxNormalizer(),
        )
        anomalies = batch_detector.detect(convert_external_ts_to_internal(timeseries))
        self._update_anomalies(timeseries, anomalies)
        return timeseries

    def _online_detect(self, alert: AlertInSeer) -> List[TimeSeriesPoint]:
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
        historic = self.alert_data_accessor.query(alert.id)
        if historic is None:
            raise Exception(f"Invalid alert id {alert.id}")

        # TODO: Need to check the time gap between historic data and the new datapoint against the alert configuration

        # Run batch detect on history data
        # TODO: This step can be optimized further by caching the matrix profile in the database
        mp_config = MPConfig(ignore_trivial=True, normalize_mp=False)
        batch_detector = MPBatchAnomalyDetector(
            config=mp_config,
            scorer=MPIRQScorer(),
            ws_selector=SuSSWindowSizeSelector(),
            normalizer=MinMaxNormalizer(),
        )
        anomalies = batch_detector.detect(historic.timeseries)

        # Run stream detection
        stream_detector = MPStreamAnomalyDetector(
            config=mp_config,
            scorer=MPIRQScorer(),
            normalizer=MinMaxNormalizer(),
            base_timestamps=historic.timeseries.timestamps,
            base_values=historic.timeseries.values,
            base_mp=anomalies.matrix_profile,
            window_size=anomalies.window_size,
        )
        streamed_anomalies = stream_detector.detect(convert_external_ts_to_internal(ts_external))
        self._update_anomalies(ts_external, streamed_anomalies)

        # Save new data point
        self.alert_data_accessor.save_timepoint(
            external_alert_id=alert.id, timepoint=ts_external[0]
        )
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

    def store_data(self, request: StoreDataRequest) -> StoreDataResponse:

        logger.info(
            "store_alert_request",
            extra={
                "organization_id": request.organization_id,
                "project_id": request.project_id,
                "external_alert_id": request.alert.id,
            },
        )
        self.alert_data_accessor.save_alert(
            organization_id=request.organization_id,
            project_id=request.project_id,
            external_alert_id=request.alert.id,
            config=request.config,
            timeseries=request.timeseries,
        )
        return StoreDataResponse(success=True)
