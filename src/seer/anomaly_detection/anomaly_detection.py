import datetime
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sentry_sdk
import stumpy  # type: ignore # mypy throws "missing library stubs"
from datadog.dogstatsd.base import statsd
from pydantic import BaseModel
from stumpy import cache

from seer.anomaly_detection.accessors import AlertDataAccessor, DbAlertDataAccessor
from seer.anomaly_detection.anomaly_detection_di import anomaly_detection_module
from seer.anomaly_detection.detectors import MPBatchAnomalyDetector, MPStreamAnomalyDetector
from seer.anomaly_detection.detectors.prophet_anomaly_detector import ProphetAnomalyDetector
from seer.anomaly_detection.models import (
    AlertAlgorithmType,
    AlgoConfig,
    DynamicAlert,
    MPTimeSeriesAnomalies,
    MPTimeSeriesAnomaliesSingleWindow,
    TimeSeries,
)
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
        cache._enable()  # This ensures that the compiled version of stumpy is cached on disk for future use.
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
    def _batch_detect_internal(
        self,
        ts_internal: TimeSeries,
        config: AnomalyDetectionConfig,
        window_size: int | None,
        algo_config: AlgoConfig,
        time_budget_ms: int | None,
        prophet_forecast_len_days: float | None = None,
    ) -> Tuple[MPTimeSeriesAnomaliesSingleWindow, pd.DataFrame]:
        logger.info(
            f"Detecting anomalies for time series with {len(ts_internal.timestamps)} datapoints"
        )
        batch_detector = MPBatchAnomalyDetector()
        prophet_detector = ProphetAnomalyDetector()
        forecast_len = (
            prophet_forecast_len_days * 24 * 60 // config.time_period
            if prophet_forecast_len_days
            else algo_config.prophet_forecast_len * (60 // config.time_period)
        )
        forecast_len = int(forecast_len)
        prophet_df = prophet_detector.predict(
            ts_internal.timestamps,
            ts_internal.values,
            forecast_len,
            config.time_period,
            config.sensitivity,
        )
        prophet_df.ds = prophet_df.ds.astype(int) / 10**9
        anomalies = batch_detector.detect(
            ts_internal,
            config,
            algo_config=algo_config,
            window_size=window_size,
            prophet_df=prophet_df,
            time_budget_ms=time_budget_ms,
        )
        return anomalies, prophet_df

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

        ts_internal = convert_external_ts_to_internal(timeseries)
        anomalies, prophet_df = self._batch_detect_internal(
            ts_internal=ts_internal,
            config=config,
            window_size=window_size,
            algo_config=algo_config,
            time_budget_ms=time_budget_ms,
        )
        batch_anomalies = DbAlertDataAccessor().combine_anomalies(
            anomalies, None, [True] * len(anomalies.flags)
        )

        return timeseries, batch_anomalies, prophet_df

    @sentry_sdk.trace
    def _save_and_fire_pruning_task(
        self,
        timepoint: TimeSeriesPoint,
        streamed_anomalies: MPTimeSeriesAnomaliesSingleWindow,
        algo_data: Optional[dict],
        historic: DynamicAlert,
        config: AnomalyDetectionConfig,
        alert_data_accessor: AlertDataAccessor,
        queue_cleanup_predict_task: bool,
        force_cleanup: bool,
    ) -> None:
        # Save new data point
        alert_data_accessor.save_timepoint(
            external_alert_id=historic.external_alert_id,
            timepoint=timepoint,
            anomaly=streamed_anomalies,
            anomaly_algo_data=algo_data,
        )

        if queue_cleanup_predict_task:
            # Delayed import due to circular imports
            from seer.anomaly_detection.tasks import cleanup_timeseries_and_predict

            try:
                # Set flag and create new task for cleanup if we have enough history and too many old points or not enough predictions remaining
                cleanup_predict_config = historic.cleanup_predict_config
                if alert_data_accessor.can_queue_cleanup_predict_task(
                    historic.external_alert_id,
                    apply_time_threshold=not force_cleanup,  # if we are forcing cleanup, we should ignore the time threshold check
                ) and (
                    force_cleanup
                    or (
                        cleanup_predict_config.num_old_points
                        >= cleanup_predict_config.num_acceptable_points
                        or cleanup_predict_config.num_predictions_remaining
                        <= cleanup_predict_config.num_acceptable_predictions
                    )
                ):
                    alert_data_accessor.queue_data_purge_flag(historic.external_alert_id)
                    cleanup_timeseries_and_predict.apply_async(
                        (historic.external_alert_id, cleanup_predict_config.timestamp_threshold),
                        countdown=random.randint(
                            0, config.time_period * 45
                        ),  # Wait between 0 - time_period * 45 seconds before queuing so the tasks are not all queued at the same time and stll have a chance to run before nex detection call
                    )
            except Exception as e:
                # Reset task and capture exception
                alert_data_accessor.reset_cleanup_predict_task(historic.external_alert_id)
                sentry_sdk.capture_exception(e)
                logger.exception(e)

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
            # If there is not enough data, we will not detect any anomalies. We still should save the timepoint
            # so that we can detect anomalies in the future.
            self._save_and_fire_pruning_task(
                timepoint=ts_external[0],
                streamed_anomalies=MPTimeSeriesAnomaliesSingleWindow(
                    flags=["none"] * len(ts_external),
                    scores=[0.0] * len(ts_external),
                    thresholds=[],
                    matrix_profile=np.array([]),
                    window_size=3,
                    original_flags=[],
                    confidence_levels=[],
                    algorithm_types=[],
                ),
                algo_data=None,
                historic=historic,
                config=config,
                alert_data_accessor=alert_data_accessor,
                queue_cleanup_predict_task=(
                    len(historic.timeseries.timestamps) == min_data - 1
                ),  # we queue cleanup task if we have just enough data to detect anomalies in the next call
                force_cleanup=True,
            )
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

        # Original Combined Flags represent the original flags AFTER mp and prophet flags have been combined but BEFORE smoothing
        original_combined_flags = [
            "anomaly_higher_confidence" if algorithm_type != AlertAlgorithmType.NONE else "none"
            for algorithm_type in historic.anomalies.algorithm_types
        ]

        # Run stream detection
        # SuSS Window
        stream_detector = MPStreamAnomalyDetector(
            history_timestamps=historic.timeseries.timestamps,
            history_values=historic.timeseries.values,
            history_mp=anomalies.matrix_profile_suss,
            window_size=anomalies.window_size,
            original_flags=original_flags,
            original_combined_flags=original_combined_flags,
        )
        streamed_anomalies = stream_detector.detect(
            convert_external_ts_to_internal(ts_external),
            config,
            prophet_df=historic.prophet_predictions.as_dataframe(),
        )

        # Commenting out in case we need to include dynamic window logic again
        # streamed_anomalies_fixed = None
        # if not historic.only_suss:
        #     # Fixed Window
        #     stream_detector_fixed = MPStreamAnomalyDetector(
        #         history_timestamps=historic.timeseries.timestamps,
        #         history_values=historic.timeseries.values,
        #         history_mp=anomalies.matrix_profile_fixed,
        #         window_size=10,
        #         original_flags=original_flags,
        #     )
        #     streamed_anomalies_fixed = stream_detector_fixed.detect(
        #         convert_external_ts_to_internal(ts_external), config
        #     )

        #     # Check if next detection should switch window
        #     use_suss_window = anomalies.use_suss[-1]
        #     if use_suss_window and streamed_anomalies_suss.flags[-1] == "anomaly_higher_confidence":
        #         use_suss_window = False

        #     # If we are using fixed window and we are past the SuSS anomalous region
        #     if (
        #         not use_suss_window
        #         and streamed_anomalies_fixed.flags[-1] == "none"
        #         and streamed_anomalies_suss.flags[-1] == "none"
        #     ):
        #         use_suss_window = True

        #     anomalies.use_suss.append(use_suss_window)

        # else:
        #     anomalies.use_suss.append(True)
        anomalies.use_suss.append(True)

        num_anomlies = len(streamed_anomalies.flags)
        streamed_anomalies_online = alert_data_accessor.combine_anomalies(
            streamed_anomalies, None, anomalies.use_suss[-num_anomlies:]
        )
        self._save_and_fire_pruning_task(
            timepoint=ts_external[0],
            streamed_anomalies=streamed_anomalies,
            algo_data=streamed_anomalies_online.get_anomaly_algo_data(len(ts_external))[0],
            historic=historic,
            config=config,
            alert_data_accessor=alert_data_accessor,
            queue_cleanup_predict_task=True,
            force_cleanup=False,
        )

        return ts_external, streamed_anomalies_online

    def _min_required_timesteps(self, time_period, min_num_days=7):
        return int(min_num_days * 24 * 60 / time_period)

    def _shift_current_by(
        self, history: TimeSeries, current: TimeSeries, num_rows: int, inplace: bool = False
    ):
        if inplace:
            history.values = np.append(history.values, current.values[0:num_rows])
            history.timestamps = np.append(history.timestamps, current.timestamps[0:num_rows])
            current.values = current.values[num_rows:]
            current.timestamps = current.timestamps[num_rows:]
            return history, current
        else:
            new_history = history.model_copy(deep=True)
            new_current = current.model_copy(deep=True)
            return self._shift_current_by(new_history, new_current, num_rows, inplace=True)

    @sentry_sdk.trace
    @inject
    def _combo_detect(
        self,
        ts_with_history: TimeSeriesWithHistory,
        ad_config: AnomalyDetectionConfig,
        time_budget_ms: int | None = None,
        algo_config: AlgoConfig = injected,
    ) -> Tuple[List[TimeSeriesPoint], MPTimeSeriesAnomalies]:
        """
        Stateless online anomaly detection for a part of a time series. This function takes two parts of the time series -
        historic time steps and current time steps. Each time step in the current section is evaluated in a streaming fashion
        against the historic data

        Parameters:
        ts_with_history: TimeSeriesWithHistory
            A full time series split into history and current

        ad_config: AnomalyDetectionConfig
            Parameters for tweaking the AD algorithm

        Returns:
        Tuple with input timeseries and identified anomalies
        """
        min_len = self._min_required_timesteps(ad_config.time_period)
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
            f"Detecting anomalies for time series with history of {len(ts_with_history.history)} datapoints and current of {orig_curr_len} datapoints"
        )

        historic = convert_external_ts_to_internal(ts_with_history.history)
        current = convert_external_ts_to_internal(ts_with_history.current)

        trim_current_by = 0
        time_period = ad_config.time_period
        num_rows_per_day = (24 * 60) // time_period
        max_stream_detection_len = int(
            algo_config.max_stream_days_for_combo_detection[time_period] * num_rows_per_day
        )
        max_batch_detection_len = int(
            algo_config.max_batch_days_for_combo_detection[time_period] * num_rows_per_day
        )
        if orig_curr_len > max_stream_detection_len:
            trim_current_by = orig_curr_len - max_stream_detection_len
            logger.info(
                f"Limiting stream detection to last {max_stream_detection_len} datapoints of the original {orig_curr_len} datapoints."
            )
            historic, current = self._shift_current_by(
                history=historic, current=current, num_rows=trim_current_by
            )

        if len(historic.values) > max_batch_detection_len:
            trim_historic_by = len(historic.values) - max_batch_detection_len
            logger.info(
                f"Limiting batch detection to last {max_batch_detection_len} datapoints of the original {len(historic.values)} datapoints."
            )
            historic.values = historic.values[trim_historic_by:]
            historic.timestamps = historic.timestamps[trim_historic_by:]

        time_allocated = datetime.timedelta(milliseconds=time_budget_ms) if time_budget_ms else None
        time_start = datetime.datetime.now(datetime.timezone.utc)

        agg_streamed_anomalies = MPTimeSeriesAnomaliesSingleWindow(
            flags=[],
            scores=[],
            thresholds=[],
            matrix_profile=np.array([]),
            window_size=100,
            original_flags=[],
            confidence_levels=[],
            algorithm_types=[],
        )

        historic_anomalies, prophet_df = self._batch_detect_internal(
            ts_internal=historic,
            config=ad_config,
            window_size=None,
            time_budget_ms=time_budget_ms,
            algo_config=algo_config,
            prophet_forecast_len_days=max_stream_detection_len,  # This ensures that the there is only prophet detection run always
        )
        if time_budget_ms is not None:
            time_elapsed = datetime.datetime.now(datetime.timezone.utc) - time_start
            time_budget_ms = time_budget_ms - (time_elapsed.microseconds // 1000)
            if time_budget_ms < 100:  # need atleast 100 milliseconds for stream detection
                logger.error(
                    "batch_detection_took_too_long",
                    extra={"time_taken": time_elapsed, "time_allocated": time_allocated},
                )
                raise ServerError("Batch detection took too long")

        agg_streamed_anomalies = MPTimeSeriesAnomaliesSingleWindow(
            flags=historic_anomalies.flags[-trim_current_by:],
            scores=historic_anomalies.scores[-trim_current_by:],
            thresholds=(
                historic_anomalies.thresholds[-trim_current_by:]
                if historic_anomalies.thresholds
                else None
            ),
            matrix_profile=historic_anomalies.matrix_profile[-trim_current_by:],
            window_size=historic_anomalies.window_size,
            original_flags=historic_anomalies.original_flags[-trim_current_by:],
            confidence_levels=historic_anomalies.confidence_levels[-trim_current_by:],
            algorithm_types=historic_anomalies.algorithm_types[-trim_current_by:],
        )
        original_combined_flags = [
            "anomaly_higher_confidence" if algorithm_type != AlertAlgorithmType.NONE else "none"
            for algorithm_type in historic_anomalies.algorithm_types
        ]
        stream_detector = MPStreamAnomalyDetector(
            history_timestamps=historic.timestamps,
            history_values=historic.values,
            history_mp=historic_anomalies.matrix_profile,
            window_size=historic_anomalies.window_size,
            original_flags=historic_anomalies.original_flags,
            original_combined_flags=original_combined_flags,
        )
        num_points_to_stream = min(max_stream_detection_len, len(current.values))
        cur_stream_ts = TimeSeries(
            timestamps=current.timestamps[0:num_points_to_stream],
            values=current.values[0:num_points_to_stream],
        )

        streamed_anomalies = stream_detector.detect(
            cur_stream_ts,
            ad_config,
            time_budget_ms=time_budget_ms,
            prophet_df=prophet_df,
        )
        agg_streamed_anomalies = agg_streamed_anomalies.extend(streamed_anomalies)

        # Ensure safe slicing by using minimum length for arrays
        safe_orig_curr_len = min(
            orig_curr_len, len(agg_streamed_anomalies.flags), len(agg_streamed_anomalies.scores)
        )

        final_streamed_anomalies = MPTimeSeriesAnomaliesSingleWindow(
            flags=agg_streamed_anomalies.flags[-safe_orig_curr_len:],
            scores=agg_streamed_anomalies.scores[-safe_orig_curr_len:],
            thresholds=(
                agg_streamed_anomalies.thresholds[-safe_orig_curr_len:]
                if agg_streamed_anomalies.thresholds
                else None
            ),
            matrix_profile=agg_streamed_anomalies.matrix_profile[-safe_orig_curr_len:],
            window_size=agg_streamed_anomalies.window_size,
            original_flags=agg_streamed_anomalies.original_flags[-safe_orig_curr_len:],
            confidence_levels=agg_streamed_anomalies.confidence_levels[-safe_orig_curr_len:],
            algorithm_types=agg_streamed_anomalies.algorithm_types[-safe_orig_curr_len:],
        )

        converted_anomalies = DbAlertDataAccessor().combine_anomalies(
            final_streamed_anomalies, None, [True] * safe_orig_curr_len
        )

        # Make sure we're not returning more points than we have anomaly data for
        return ts_with_history.current[:safe_orig_curr_len], converted_anomalies

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
        self, request: DetectAnomaliesRequest, time_budget_ms: int = 9500
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
            with statsd.timed("seer.anomaly_detection.detect.online_detect_duration"):
                statsd.increment("seer.anomaly_detection.detect.online_detect")
                sentry_sdk.set_tag(AnomalyDetectionTags.ALERT_ID, request.context.id)
                ts, anomalies = self._online_detect(request.context, request.config)
        elif isinstance(request.context, TimeSeriesWithHistory):
            with statsd.timed("seer.anomaly_detection.detect.combo_detect_duration"):
                statsd.increment("seer.anomaly_detection.detect.combo_detect")
                ts, anomalies = self._combo_detect(
                    request.context, request.config, time_budget_ms=time_budget_ms
                )
        else:
            with statsd.timed("seer.anomaly_detection.detect.batch_detect_duration"):
                statsd.increment("seer.anomaly_detection.detect.batch_detect")
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
            logger.error(
                "batch_detection_took_too_long",
                extra={"time_taken": time_elapsed, "time_allocated": time_allocated},
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
