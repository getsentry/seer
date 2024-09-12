import logging
from typing import List, Tuple

import numpy as np
import sentry_sdk
import stumpy  # type: ignore # mypy throws "missing library stubs"
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
from seer.exceptions import ClientError, ServerError

anomaly_detection_module.enable()
logger = logging.getLogger(__name__)


class AnomalyDetection(BaseModel):
    @sentry_sdk.trace
    def __init__(self):
        """
        Force Stumpy compilation by making a dummy call to the library.

        Note: compilation triggered here is very specific to the exact values of parameters ignore_trivial, normalize etc. A
            future call with different values for one or more parameter will still trigger a recompilation.
        """
        data = np.arange(10.0)
        mp = stumpy.stump(data, m=3, ignore_trivial=True, normalize=False)
        stream = stumpy.stumpi(
            data,
            m=3,
            mp=mp,
            normalize=False,
            egress=False,
        )
        stream.update(6.0)

    @sentry_sdk.trace
    def _batch_detect(
        self, timeseries: List[TimeSeriesPoint], config: AnomalyDetectionConfig
    ) -> Tuple[List[TimeSeriesPoint], MPTimeSeriesAnomalies]:
        """
        Stateless batch anomaly detection on entire timeseries as provided. In batch mode, analysis of a
        single timestep is dependent on the time steps on either side of it.

        Parameters:
        timeseries: TimeSeries
            The full timeseries

        config: AnomalyDetectionConfig
            Parameters for tweaking the AD algorithm

        Returns:
        Tuple with input timeseries and identified anomalies
        """
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
        """
        Online anomaly detection for a new incoming timestep of an existing alert. In this mode, the new timestep is
        analyzed using historic timesteps from the datastore

        Parameters:
        alert: AlertInSeer
            Alert id as well as the new time step to evaluate

        config: AnomalyDetectionConfig
            Parameters for tweaking the AD algorithm

        Returns:
        Tuple with input timeseries and identified anomalies
        """

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

            logger.error(
                "no_stored_history_data",
                extra={
                    "alert_id": alert.id,
                },
            )
            raise ClientError("No timeseries data found for alert")

        if not isinstance(historic.anomalies, MPTimeSeriesAnomalies):
            logger.error(
                "invalid_state",
                extra={
                    "note": f"Expecting object of type MPTimeSeriesAnomalies but found {type(historic.anomalies)}"
                },
            )
            raise ServerError("Invalid state")
        anomalies: MPTimeSeriesAnomalies = historic.anomalies

        # TODO: Need to check the time gap between historic data and the new datapoint against the alert configuration

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
            anomaly_algo_data=streamed_anomalies.get_anomaly_algo_data(len(ts_external))[0],
        )
        # TODO: Clean up old data
        return ts_external, streamed_anomalies

    def _min_required_timesteps(self, time_period, min_num_days=7):
        return int(min_num_days * 24 * 60 / time_period)

    @sentry_sdk.trace
    def _combo_detect(
        self, ts_with_history: TimeSeriesWithHistory, config: AnomalyDetectionConfig
    ) -> Tuple[List[TimeSeriesPoint], MPTimeSeriesAnomalies]:
        """
        Stateless online anomaly detection for a part of a time series. This function takes two parts of the time series -
        historic time steps and current time steps. Each time step in the current section is evaluated in a streaming fashion
        against the historic data

        Parameters:
        ts_with_history: TimeSeriesWithHistory
            A full time series split into history and current

        config: AnomalyDetectionConfig
            Parameters for tweaking the AD algorithm

        Returns:
        Tuple with input timeseries and identified anomalies
        """

        min_len = self._min_required_timesteps(config.time_period)
        if len(ts_with_history.history) < min_len:
            logger.error(
                "insufficient_history_data",
                extra={
                    "num_datapoints": len(ts_with_history.history),
                    "minimum_required": min_len,
                },
            )
            raise ClientError("Insufficient history data")

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
        for i, point in enumerate(ts_external):
            point.anomaly = Anomaly(
                anomaly_score=anomalies.scores[i],
                anomaly_type=anomalies.flags[i],
            )

    @sentry_sdk.trace
    def detect_anomalies(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        """
        Main entry point for anomaly detection.

        Parameters:
        request: DetectAnomaliesRequest
            Anomaly detection request that has either a complete time series or an alert reference.
        """
        if isinstance(request.context, AlertInSeer):
            mode = "streaming.alert"
        elif isinstance(request.context, TimeSeriesWithHistory):
            mode = "streaming.ts_with_history"
        else:
            mode = "batch.ts_full"

        sentry_sdk.set_tag("ad_mode", mode)

        if isinstance(request.context, AlertInSeer):
            ts, anomalies = self._online_detect(request.context, request.config)
        elif isinstance(request.context, TimeSeriesWithHistory):
            ts, anomalies = self._combo_detect(request.context, request.config)
        else:
            ts, anomalies = self._batch_detect(request.context, request.config)
        self._update_anomalies(ts, anomalies)
        return DetectAnomaliesResponse(success=True, timeseries=ts)

    @inject
    def store_data(
        self, request: StoreDataRequest, alert_data_accessor: AlertDataAccessor = injected
    ) -> StoreDataResponse:
        """
        Main entry point for storing time series data for an alert.

        Parameters:
        request: StoreDataRequest
            Alert information along with underlying time series data
        """
        # Ensure we have at least 7 days of data in the time series
        min_len = self._min_required_timesteps(request.config.time_period)
        if len(request.timeseries) < min_len:
            logger.error(
                "insufficient_timeseries_data",
                extra={
                    "organization_id": request.organization_id,
                    "project_id": request.project_id,
                    "external_alert_id": request.alert.id,
                    "num_datapoints": len(request.timeseries),
                    "minimum_required": min_len,
                },
            )
            raise ClientError("Insufficient time series data for alert")

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
