import datetime
import logging
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import sentry_sdk
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel

from seer.anomaly_detection.accessors import AlertDataAccessor
from seer.anomaly_detection.anomaly_detection_di import anomaly_detection_module
from seer.anomaly_detection.detectors import MPBatchAnomalyDetector, MPStreamAnomalyDetector
from seer.anomaly_detection.detectors.prophet_anomaly_detector import ProphetAnomalyDetector
from seer.anomaly_detection.models import AlgoConfig, MPTimeSeriesAnomalies
from seer.anomaly_detection.models.converters import convert_external_ts_to_internal
from seer.anomaly_detection.models.external import (
    AlertInSeer,
    Anomaly,
    AnomalyDetectionConfig,
    DeleteAlertDataRequest,
    DeleteAlertDataResponse,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
    TimeSeriesPoint,
    TimeSeriesWithHistory,
)
from seer.anomaly_detection.models.timeseries import ProphetPrediction
from seer.db import TaskStatus
from seer.dependency_injection import inject, injected
from seer.exceptions import ClientError, ServerError
from seer.tags import AnomalyDetectionModes, AnomalyDetectionTags

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

    @inject
    @sentry_sdk.trace
    def _batch_detect(
        self,
        timeseries: List[TimeSeriesPoint],
        config: AnomalyDetectionConfig,
        window_size: int | None = None,
        algo_config: AlgoConfig = injected,
        time_budget_ms: int | None = None,
    ) -> Tuple[List[TimeSeriesPoint], MPTimeSeriesAnomalies, pd.DataFrame]:
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
        prophet_detector = ProphetAnomalyDetector()

        forecast_len = algo_config.prophet_forecast_len * (60 // config.time_period)
        ts_internal = convert_external_ts_to_internal(timeseries)
        prophet_df = prophet_detector.predict(
            ts_internal.timestamps,
            ts_internal.values,
            forecast_len,
            config.time_period,
            config.sensitivity,
        )
        prophet_df.ds = prophet_df.ds.astype(int) / 10**9
        anomalies_suss = batch_detector.detect(
            ts_internal,
            config,
            algo_config=algo_config,
            window_size=window_size,
            prophet_df=prophet_df,
            time_budget_ms=(
                time_budget_ms // 2 if time_budget_ms else None
            ),  # Time budget is split between the two detection calls
        )
        anomalies_fixed = batch_detector.detect(
            convert_external_ts_to_internal(timeseries),
            config,
            algo_config=algo_config,
            window_size=algo_config.mp_fixed_window_size,
            time_budget_ms=(
                time_budget_ms // 2 if time_budget_ms else None
            ),  # Time budget is split between the two detection calls
            prophet_df=prophet_df,
        )
        anomalies = DbAlertDataAccessor().combine_anomalies(
            anomalies_suss, anomalies_fixed, [True] * len(timeseries)
        )

        return timeseries, anomalies, prophet_df

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
        if historic.data_purge_flag == TaskStatus.PROCESSING:
            logger.warning(
                "data_purge_flag_invalid",
                extra={
                    "alert_id": alert.id,
                    "data_purge_flag": historic.data_purge_flag,
                    "last_queued_at": historic.last_queued_at,
                },
            )

        # Confirm that there is enough data (after purge)
        min_data = self._min_required_timesteps(historic.config.time_period)
        if len(historic.timeseries.timestamps) < min_data:
            logger.error(f"Not enough timeseries data. At least {min_data} data points required")
            raise ClientError(
                f"Not enough timeseries data. At least {min_data} data points required"
            )
        anomalies: MPTimeSeriesAnomalies = historic.anomalies

        # Get the original flags from the historic data
        original_flags = anomalies.original_flags
        if original_flags is not None and len(original_flags) != len(
            historic.timeseries.timestamps
        ):
            logger.error(
                "invalid_state",
                extra={"note": "Original flags and timeseries are not of the same length"},
            )
            raise ServerError("Invalid state")

        # Run stream detection
        stream_detector = MPStreamAnomalyDetector(
            history_timestamps=historic.timeseries.timestamps,
            history_values=historic.timeseries.values,
            history_mp=anomalies.matrix_profile,
            window_size=anomalies.window_size,
            original_flags=original_flags,
        )
        streamed_anomalies_suss = stream_detector_suss.detect(
            convert_external_ts_to_internal(ts_external),
            config,
            prophet_df=historic.prophet_predictions.as_dataframe(),
        )

        streamed_anomalies_fixed = None
        if not historic.only_suss:
            # Fixed Window
            stream_detector_fixed = MPStreamAnomalyDetector(
                history_timestamps=historic.timeseries.timestamps,
                history_values=historic.timeseries.values,
                history_mp=anomalies.matrix_profile_fixed,
                window_size=10,
                original_flags=original_flags,
            )
            streamed_anomalies_fixed = stream_detector_fixed.detect(
                convert_external_ts_to_internal(ts_external),
                config,
                prophet_df=historic.prophet_predictions.as_dataframe(),
            )

            # Check if next detection should switch window
            use_suss_window = anomalies.use_suss[-1]
            if use_suss_window and streamed_anomalies_suss.flags[-1] == "anomaly_higher_confidence":
                use_suss_window = False

            # If we are using fixed window and we are past the SuSS anomalous region
            if (
                not use_suss_window
                and streamed_anomalies_fixed.flags[-1] == "none"
                and streamed_anomalies_suss.flags[-1] == "none"
            ):
                use_suss_window = True

            anomalies.use_suss.append(use_suss_window)

        else:
            anomalies.use_suss.append(True)

        num_anomlies = len(streamed_anomalies_suss.flags)
        streamed_anomalies = alert_data_accessor.combine_anomalies(
            streamed_anomalies_suss, streamed_anomalies_fixed, anomalies.use_suss[-num_anomlies:]
        )

        # Save new data point
        alert_data_accessor.save_timepoint(
            external_alert_id=alert.id,
            timepoint=ts_external[0],
            anomaly=streamed_anomalies,
            anomaly_algo_data=streamed_anomalies.get_anomaly_algo_data(len(ts_external))[0],
        )

        # Delayed import due to circular imports
        from seer.anomaly_detection.tasks import cleanup_timeseries_and_predict

        try:
            # Set flag and create new task for cleanup if too many old points or not enough predictions remaining
            cleanup_predict_config = historic.cleanup_predict_config
            if alert_data_accessor.can_queue_cleanup_predict_task(historic.external_alert_id) and (
                cleanup_predict_config.num_old_points
                >= cleanup_predict_config.num_acceptable_points
                or cleanup_predict_config.num_predictions_remaining
                <= cleanup_predict_config.num_acceptable_predictions
            ):
                alert_data_accessor.queue_data_purge_flag(historic.external_alert_id)
                cleanup_timeseries_and_predict.apply_async(
                    (historic.external_alert_id, cleanup_predict_config.timestamp_threshold),
                    countdown=random.randint(
                        0, config.time_period * 60
                    ),  # Wait between 0 - time_period * 60 seconds before queuing so the tasks are not all queued at the same time
                )
        except Exception as e:
            # Reset task and capture exception
            alert_data_accessor.reset_cleanup_predict_task(historic.external_alert_id)
            sentry_sdk.capture_exception(e)
            logger.exception(e)

        return ts_external, streamed_anomalies

    def _min_required_timesteps(self, time_period, min_num_days=7):
        return int(min_num_days * 24 * 60 / time_period)

    @sentry_sdk.trace
    def _combo_detect(
        self,
        ts_with_history: TimeSeriesWithHistory,
        config: AnomalyDetectionConfig,
        time_budget_ms: int | None = None,
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
            logger.warning(
                "insufficient_history_data",
                extra={
                    "num_datapoints": len(ts_with_history.history),
                    "minimum_required": min_len,
                },
            )
            raise ClientError("Insufficient history data")

        orig_curr_len = len(ts_with_history.current)
        logger.info(
            f"Detecting anomalies for time series with {len(ts_with_history.history)} datapoints and history of {orig_curr_len} datapoints"
        )
        trim_current_by = 0
        MAX_STREAM_DETECTION_LEN = 400
        if orig_curr_len > MAX_STREAM_DETECTION_LEN:
            trim_current_by = orig_curr_len - MAX_STREAM_DETECTION_LEN
            logger.info(
                f"Limiting stream detection to last 400 datapoints of the original {orig_curr_len} datapoints."
            )
            ts_with_history.history.extend(ts_with_history.current[0:-MAX_STREAM_DETECTION_LEN])
            ts_with_history.current = ts_with_history.current[-MAX_STREAM_DETECTION_LEN:]
        ts_external: List[TimeSeriesPoint] = ts_with_history.current

        historic = convert_external_ts_to_internal(ts_with_history.history)
        # We are doing 4 detection operations, so allocating 1/4th of the time budget for each one.
        time_budget_ms = time_budget_ms // 2 if time_budget_ms else None

        # Run batch detect on history data
        batch_detector = MPBatchAnomalyDetector()
        historic_anomalies_suss = batch_detector.detect(
            historic,
            config,
            time_budget_ms=time_budget_ms,
            # TODO Compute and pass prophet df?
        )
        historic_anomalies_fixed = batch_detector.detect(
            historic,
            config,
            window_size=10,
            time_budget_ms=time_budget_ms,
            # TODO Compute and pass prophet df?
        )

        # Run stream detection on current data
        stream_detector = MPStreamAnomalyDetector(
            history_timestamps=historic.timestamps,
            history_values=historic.values,
            history_mp=historic_anomalies.matrix_profile,
            window_size=historic_anomalies.window_size,
            original_flags=historic_anomalies.original_flags,
        )
        streamed_anomalies_suss = stream_detector_suss.detect(
            convert_external_ts_to_internal(ts_external),
            config,
            time_budget_ms=time_budget_ms,
            # TODO Compute and pass prophet df?
        )

        # Fixed Window
        stream_detector_fixed = MPStreamAnomalyDetector(
            history_timestamps=historic.timestamps,
            history_values=historic.values,
            history_mp=historic_anomalies_fixed.matrix_profile,
            window_size=historic_anomalies_fixed.window_size,
            original_flags=historic_anomalies_fixed.original_flags,
        )
        streamed_anomalies_fixed = stream_detector_fixed.detect(
            convert_external_ts_to_internal(ts_external),
            config,
            time_budget_ms=time_budget_ms,
            # TODO Compute and pass prophet df?
        )

        if trim_current_by > 0:
            ts_external = ts_with_history.history[-trim_current_by:] + ts_external
            streamed_anomalies.flags = (
                historic_anomalies.flags[-trim_current_by:] + streamed_anomalies.flags
            )
            streamed_anomalies.scores = (
                historic_anomalies.scores[-trim_current_by:] + streamed_anomalies.scores
            )

        return ts_external, streamed_anomalies

    def _update_anomalies(
        self,
        ts_external: List[TimeSeriesPoint],
        anomalies: MPTimeSeriesAnomalies,
    ):
        for i, point in enumerate(ts_external):
            point.anomaly = Anomaly(
                anomaly_score=anomalies.scores[i],
                anomaly_type=anomalies.flags[i],
            )

    @sentry_sdk.trace
    def detect_anomalies(
        self, request: DetectAnomaliesRequest, time_budget_ms: int = 4500
    ) -> DetectAnomaliesResponse:
        """
        Main entry point for anomaly detection.

        Parameters:
        request: DetectAnomaliesRequest
            Anomaly detection request that has either a complete time series or an alert reference.
        """
        if isinstance(request.context, AlertInSeer):
            mode = AnomalyDetectionModes.STREAMING_ALERT
        elif isinstance(request.context, TimeSeriesWithHistory):
            mode = AnomalyDetectionModes.STREAMING_TS_WITH_HISTORY
        else:
            mode = AnomalyDetectionModes.BATCH_TS_FULL

        sentry_sdk.set_tag(AnomalyDetectionTags.MODE, mode)

        if isinstance(request.context, AlertInSeer):
            sentry_sdk.set_tag(AnomalyDetectionTags.ALERT_ID, request.context.id)
            ts, anomalies = self._online_detect(request.context, request.config)
        elif isinstance(request.context, TimeSeriesWithHistory):
            ts, anomalies = self._combo_detect(
                request.context, request.config, time_budget_ms=time_budget_ms
            )
        else:
            ts, anomalies, _ = self._batch_detect(
                request.context, request.config, time_budget_ms=time_budget_ms
            )
        self._update_anomalies(ts, anomalies)
        return DetectAnomaliesResponse(success=True, timeseries=ts)

    @inject
    def store_data(
        self,
        request: StoreDataRequest,
        alert_data_accessor: AlertDataAccessor = injected,
        time_budget_ms: int = 4500,  # Allocating 4.5 seconds as alerting system timesout after 5 seconds
    ) -> StoreDataResponse:
        """
        Main entry point for storing time series data for an alert.

        Parameters:
        request: StoreDataRequest
            Alert information along with underlying time series data
        """
        sentry_sdk.set_tag(AnomalyDetectionTags.ALERT_ID, request.alert.id)
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
                    "config": request.config.model_dump(),
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
                "config": request.config.model_dump(),
            },
        )
        time_start = datetime.datetime.now()
        ts, anomalies, prophet_df = self._batch_detect(
            request.timeseries, request.config, time_budget_ms=time_budget_ms
        )
        time_elapsed = datetime.datetime.now() - time_start
        time_allocated = datetime.timedelta(milliseconds=time_budget_ms)
        if time_elapsed > time_allocated:
            sentry_sdk.set_extra("time_taken_for_batch_detection", time_elapsed)
            sentry_sdk.set_extra("time_allocated_for_batch_detection", time_allocated)
            sentry_sdk.capture_message(
                "batch_detection_took_too_long",
                level="error",
            )
            raise ServerError(
                "Batch detection took too long"
            )  # Abort without saving to avoid data going out of sync with alerting system.

        saved_alert_id = alert_data_accessor.save_alert(
            organization_id=request.organization_id,
            project_id=request.project_id,
            external_alert_id=request.alert.id,
            config=request.config,
            timeseries=ts,
            anomalies=anomalies,
            anomaly_algo_data={"window_size": anomalies.window_size},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        alert_data_accessor.store_prophet_predictions(
            saved_alert_id, ProphetPrediction.from_prophet_df(prophet_df)
        )
        return StoreDataResponse(success=True)

    @inject
    def delete_alert_data(
        self, request: DeleteAlertDataRequest, alert_data_accessor: AlertDataAccessor = injected
    ) -> DeleteAlertDataResponse:
        """
        Main entry point for deleting data related to an alert.

        Parameters:
        request: DeleteAlertDataRequest
            Alert to clear
        """
        sentry_sdk.set_tag(AnomalyDetectionTags.ALERT_ID, request.alert.id)
        alert_data_accessor.delete_alert_data(external_alert_id=request.alert.id)
        return DeleteAlertDataResponse(success=True)
