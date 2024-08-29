import logging
from typing import List, Tuple

import sentry_sdk
from pydantic import BaseModel

from seer.anomaly_detection.accessors import AlertDataAccessor
from seer.anomaly_detection.anomaly_detection_di import anomaly_detection_module
from seer.anomaly_detection.detectors import MPBatchAnomalyDetector, MPStreamAnomalyDetector
from seer.anomaly_detection.models import MPTimeSeriesAnomalies, TimeSeriesAnomalies
from seer.anomaly_detection.models.converters import convert_external_ts_to_internal
from seer.anomaly_detection.models.external import (
    AlertInSeer,
    Anomaly,
    AnomalyDetectionConfig,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
    TimeSeriesPoint,
    TimeSeriesWithHistory,
)
from seer.dependency_injection import inject, injected

anomaly_detection_module.enable()
logger = logging.getLogger(__name__)


class AnomalyDetection(BaseModel):
    @sentry_sdk.trace
    def _batch_detect(
        self, timeseries: List[TimeSeriesPoint], config: AnomalyDetectionConfig
    ) -> Tuple[List[TimeSeriesPoint], MPTimeSeriesAnomalies]:
        logger.info(f"Detecting anomalies for time series with {len(timeseries)} datapoints")
        batch_detector = MPBatchAnomalyDetector()
        anomalies = batch_detector.detect(convert_external_ts_to_internal(timeseries), config)
        return timeseries, anomalies

    @inject
    @sentry_sdk.trace
    def _online_detect(
        self,
        alert: AlertInSeer,
        config: AnomalyDetectionConfig,
        alert_data_accessor: AlertDataAccessor = injected,
    ) -> Tuple[List[TimeSeriesPoint], MPTimeSeriesAnomalies]:
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
        anomalies = batch_detector.detect(historic.timeseries, config)

        # Run stream detection
        stream_detector = MPStreamAnomalyDetector(
            base_timestamps=historic.timeseries.timestamps,
            base_values=historic.timeseries.values,
            base_mp=anomalies.matrix_profile,
            window_size=anomalies.window_size,
        )
        streamed_anomalies = stream_detector.detect(
            convert_external_ts_to_internal(ts_external), config
        )

        # Save new data point
        alert_data_accessor.save_timepoint(
            external_alert_id=alert.id,
            timepoint=ts_external[0],
            anomaly=streamed_anomalies,
            anomaly_algo_data=streamed_anomalies.get_anomaly_algo_data()[-1],
        )
        # TODO: Clean up old data
        return ts_external, streamed_anomalies

    @inject
    @sentry_sdk.trace
    def _combo_detect(
        self, ts_with_history: TimeSeriesWithHistory, config: AnomalyDetectionConfig
    ) -> Tuple[List[TimeSeriesPoint], MPTimeSeriesAnomalies]:
        if len(ts_with_history.history) < int(7 * 24 * 60 / config.time_period):
            logger.error(
                "insufficient_history_data",
                extra={
                    "num_datapoints": len(ts_with_history.history),
                    "minimum_required": int(7 * 24 * 50 / config.time_period),
                },
            )
            raise Exception("Insufficient history data")

        logger.info(
            f"Detecting anomalies for time series with {len(ts_with_history.current)} datapoints and history of {len(ts_with_history.history)} datapoints"
        )
        ts_external: List[TimeSeriesPoint] = ts_with_history.current

        historic = convert_external_ts_to_internal(ts_with_history.history)

        # Run batch detect on history data
        batch_detector = MPBatchAnomalyDetector()
        anomalies = batch_detector.detect(historic, config)

        # Run stream detection on current data
        stream_detector = MPStreamAnomalyDetector(
            base_timestamps=historic.timestamps,
            base_values=historic.values,
            base_mp=anomalies.matrix_profile,
            window_size=anomalies.window_size,
        )
        streamed_anomalies = stream_detector.detect(
            convert_external_ts_to_internal(ts_external), config
        )
        return ts_external, streamed_anomalies

    def _update_anomalies(self, ts_external: List[TimeSeriesPoint], anomalies: TimeSeriesAnomalies):
        if anomalies is None:
            raise Exception("No anomalies available for the timeseries.")
        for i, point in enumerate(ts_external):
            point.anomaly = Anomaly(
                anomaly_score=anomalies.scores[i],
                anomaly_type=anomalies.flags[i],
            )

    def detect_anomalies(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        if isinstance(request.context, AlertInSeer):
            ts, anomalies = self._online_detect(request.context, request.config)
        elif isinstance(request.context, TimeSeriesWithHistory):
            ts, anomalies = self._combo_detect(request.context, request.config)
        else:
            ts, anomalies = self._batch_detect(request.context, request.config)
        self._update_anomalies(ts, anomalies)
        return DetectAnomaliesResponse(timeseries=ts)

    @inject
    def store_data(
        self, request: StoreDataRequest, alert_data_accessor: AlertDataAccessor = injected
    ) -> StoreDataResponse:
        # Ensure we have at least 7 days of data in the time series
        if len(request.timeseries) < int(7 * 24 * 60 / request.config.time_period):
            logger.error(
                "insufficient_timeseries_data",
                extra={
                    "organization_id": request.organization_id,
                    "project_id": request.project_id,
                    "external_alert_id": request.alert.id,
                    "num_datapoints": len(request.timeseries),
                    "minimum_required": int(7 * 24 * 50 / request.config.time_period),
                },
            )
            raise Exception(f"Insufficient time series data for alert {request.alert.id}")

        logger.info(
            "store_alert_request",
            extra={
                "organization_id": request.organization_id,
                "project_id": request.project_id,
                "external_alert_id": request.alert.id,
                "num_datapoints": len(request.timeseries),
            },
        )
        ts, anomalies = self._batch_detect(request.timeseries, request.config)
        alert_data_accessor.save_alert(
            organization_id=request.organization_id,
            project_id=request.project_id,
            external_alert_id=request.alert.id,
            config=request.config,
            timeseries=ts,
            anomalies=anomalies,
            anomaly_algo_data={"window_size": anomalies.window_size},
        )
        return StoreDataResponse(success=True)
