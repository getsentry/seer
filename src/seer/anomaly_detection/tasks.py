import logging
from datetime import datetime, timedelta
from operator import and_, or_
from typing import List

import numpy as np
import sentry_sdk
import stumpy  # type: ignore # mypy throws "missing library stubs"
from sqlalchemy import delete

from celery_app.app import celery_app
from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.detectors.anomaly_detectors import MPBatchAnomalyDetector
from seer.anomaly_detection.detectors.prophet_anomaly_detector import ProphetAnomalyDetector
from seer.anomaly_detection.models import AlgoConfig, ProphetPrediction, TimeSeries
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.db import (
    DbDynamicAlert,
    DbDynamicAlertTimeSeries,
    DbDynamicAlertTimeSeriesHistory,
    DbProphetAlertTimeSeries,
    DbProphetAlertTimeSeriesHistory,
    Session,
    TaskStatus,
)
from seer.dependency_injection import inject, injected
from seer.tags import AnomalyDetectionTags

logger = logging.getLogger(__name__)


@sentry_sdk.trace
def _init_stumpy():
    dummy_data = np.arange(10.0)
    dummy_mp = stumpy.stump(dummy_data, m=3, ignore_trivial=True, normalize=False)
    dummy_stream = stumpy.stumpi(
        dummy_data,
        m=3,
        mp=dummy_mp,
        normalize=False,
        egress=False,
    )
    dummy_stream.update(6.0)


@celery_app.task
@sentry_sdk.trace
def cleanup_timeseries_and_predict(
    external_alert_id: int | None,
    external_alert_source_id: int | None,
    external_alert_source_type: int | None,
    date_threshold: float,
):  # TODO: Add support for the new external alert id and source id
    sentry_sdk.set_tag(AnomalyDetectionTags.SEER_FUNCTIONALITY, "anomaly_detection")
    span = sentry_sdk.get_current_span()

    if span is not None:
        span.set_tag("alert_id", external_alert_id)
        span.set_tag("alert_source_id", external_alert_source_id)
        span.set_tag("alert_source_type", external_alert_source_type)

    logger.info(
        "Deleting timeseries points over 28 days old and updating matrix profiles",
        extra={
            "alert_id": external_alert_id,
            "alert_source_id": external_alert_source_id,
            "alert_source_type": external_alert_source_type,
        },
    )

    # Perform a dummy call to Stumpy to force compilation
    _init_stumpy()

    _toggle_data_purge_flag(external_alert_id, external_alert_source_id, external_alert_source_type)

    with Session() as session:
        db_accessor = DbAlertDataAccessor()
        alert = db_accessor.query_db_alert_or_raise(
            session=session,
            external_alert_id=external_alert_id,
            external_alert_source_id=external_alert_source_id,
            external_alert_source_type=external_alert_source_type,
        )

        if len(alert.timeseries) == 0:
            logger.warning(
                f"Alert with id {external_alert_id} and source id {external_alert_source_id} and source type {external_alert_source_type} has no timeseries"
            )
        else:
            config = AnomalyDetectionConfig(
                time_period=alert.config["time_period"],
                sensitivity=alert.config["sensitivity"],
                direction=alert.config["direction"],
                expected_seasonality=alert.config["expected_seasonality"],
            )

            # Delete old timeseries and prophet prediction points
            deleted_timeseries_points, prophet_deleted_timeseries_points = (
                _delete_old_timeseries_points(alert, date_threshold)
            )

            if len(alert.timeseries) > 0:
                updated_timeseries_points = _update_matrix_profiles(alert, config)
                predictions = _fit_predict(alert, config)
                db_accessor.store_prophet_predictions(alert.id, predictions)
            else:
                # Reset the window size to 0 if there are no timeseries points left
                alert.anomaly_algo_data = {"window_size": 0}
                logger.warn(
                    f"Alert with id {external_alert_id} and source id {external_alert_source_id} and source type {external_alert_source_type} has empty timeseries data after pruning"
                )
                updated_timeseries_points = 0

            session.commit()
            logger.info(
                f"Deleted {deleted_timeseries_points} timeseries points and {prophet_deleted_timeseries_points} prophet prediction points in alertd id {external_alert_id} and source id {external_alert_source_id} and source type {external_alert_source_type}"
            )
            logger.info(
                f"Updated matrix profiles for {updated_timeseries_points} points in alertd id {external_alert_id} and source id {external_alert_source_id} and source type {external_alert_source_type}"
            )

    _toggle_data_purge_flag(external_alert_id, external_alert_source_id, external_alert_source_type)


@sentry_sdk.trace
def _delete_old_timeseries_points(alert: DbDynamicAlert, date_threshold: float):

    time_series_to_remove = []
    for ts in alert.timeseries:
        if ts.timestamp.timestamp() < date_threshold:
            time_series_to_remove.append(ts)

    prophet_time_series_to_remove = []
    for prophet_ts in alert.prophet_predictions:
        if prophet_ts.timestamp.timestamp() < date_threshold:
            prophet_time_series_to_remove.append(prophet_ts)

    # Save history records before removing
    _save_timeseries_history(alert, time_series_to_remove, prophet_time_series_to_remove)

    deleted_count = 0
    for ts in time_series_to_remove:
        alert.timeseries.remove(ts)
        deleted_count += 1

    prophet_deleted_count = 0
    for prophet_ts in prophet_time_series_to_remove:
        alert.prophet_predictions.remove(prophet_ts)
        prophet_deleted_count += 1

    return deleted_count, prophet_deleted_count


@sentry_sdk.trace
def _save_timeseries_history(
    alert: DbDynamicAlert,
    timeseries: List[DbDynamicAlertTimeSeries],
    prophet_timeseries: List[DbProphetAlertTimeSeries],
):

    with Session() as session:
        for ts in timeseries:
            history_record = DbDynamicAlertTimeSeriesHistory(
                alert_id=alert.external_alert_id,
                external_alert_source_id=alert.external_alert_source_id,
                external_alert_source_type=alert.external_alert_source_type,
                timestamp=ts.timestamp,
                anomaly_type=ts.anomaly_type,
                value=ts.value,
                saved_at=datetime.now(),
            )
            session.add(history_record)

        for prophet_ts in prophet_timeseries:
            prophet_history_record = DbProphetAlertTimeSeriesHistory(
                alert_id=alert.external_alert_id,
                external_alert_source_id=alert.external_alert_source_id,
                external_alert_source_type=alert.external_alert_source_type,
                timestamp=prophet_ts.timestamp,
                yhat=prophet_ts.yhat,
                yhat_lower=prophet_ts.yhat_lower,
                yhat_upper=prophet_ts.yhat_upper,
                saved_at=datetime.now(),
            )
            session.add(prophet_history_record)

        session.commit()


@sentry_sdk.trace
@inject
def _update_matrix_profiles(
    alert: DbDynamicAlert,
    anomaly_detection_config: AnomalyDetectionConfig,
    algo_config: AlgoConfig = injected,
):

    timeseries = TimeSeries(
        timestamps=np.array([timestep.timestamp.timestamp() for timestep in alert.timeseries]),
        values=np.array([timestep.value for timestep in alert.timeseries]),
    )

    anomalies_suss = MPBatchAnomalyDetector()._compute_matrix_profile(
        timeseries=timeseries, ad_config=anomaly_detection_config, algo_config=algo_config
    )
    # anomalies_fixed = MPBatchAnomalyDetector()._compute_matrix_profile(
    #     timeseries=timeseries,
    #     ad_config=anomaly_detection_config,
    #     algo_config=algo_config,
    #     window_size=algo_config.mp_fixed_window_size,
    # )
    anomalies = DbAlertDataAccessor().combine_anomalies(
        anomalies_suss, None, [True] * len(timeseries.timestamps)
    )

    algo_data_map = dict(
        zip(timeseries.timestamps, anomalies.get_anomaly_algo_data(len(timeseries.timestamps)))
    )
    updated_timeseries_points = 0
    for timestep in alert.timeseries:
        timestep.anomaly_algo_data = algo_data_map[timestep.timestamp.timestamp()]
        updated_timeseries_points += 1
    alert.anomaly_algo_data = {"window_size": anomalies.window_size}
    return updated_timeseries_points


@sentry_sdk.trace
@inject
def _fit_predict(
    alert: DbDynamicAlert,
    config: AnomalyDetectionConfig,
    algo_config: AlgoConfig = injected,
) -> ProphetPrediction:

    prophet_detector = ProphetAnomalyDetector()

    timestamps = np.array([0.0] * len(alert.timeseries))
    values = np.array([0.0] * len(alert.timeseries))
    for i, ts in enumerate(alert.timeseries):
        timestamps[i] = ts.timestamp.timestamp()
        values[i] = ts.value

    # Create prophet_forecast_len hours worth of predictions
    forecast_len = algo_config.prophet_forecast_len * (60 // config.time_period)
    prediction_df = prophet_detector.predict(
        timestamps, values, forecast_len, config.time_period, config.sensitivity
    )
    return ProphetPrediction.from_prophet_df(prediction_df)


@sentry_sdk.trace
def _toggle_data_purge_flag(
    alert_id: int | None,
    external_alert_source_id: int | None,
    external_alert_source_type: int | None,
):

    with Session() as session:
        alert = DbAlertDataAccessor().query_db_alert_or_none(
            session=session,
            external_alert_id=alert_id,
            external_alert_source_id=external_alert_source_id,
            external_alert_source_type=external_alert_source_type,
        )

        if alert is None:
            raise ValueError(
                f"Alert with id {alert_id} and source id {external_alert_source_id} and source type {external_alert_source_type} not found"
            )
        new_flag = (
            TaskStatus.PROCESSING
            if alert.data_purge_flag == TaskStatus.QUEUED
            else TaskStatus.NOT_QUEUED
        )
        alert.data_purge_flag = new_flag
        session.commit()


@celery_app.task
@sentry_sdk.trace
def cleanup_disabled_alerts():
    sentry_sdk.set_tag(AnomalyDetectionTags.SEER_FUNCTIONALITY, "anomaly_detection")
    date_threshold = datetime.now() - timedelta(days=28)

    logger.info(
        "Cleaning up alerts inactive since 28 days ago (detection has not been run)",
        extra={"date_threshold": date_threshold},
    )

    with Session() as session:
        # Get and delete alerts that haven't been queued for detection in the last 28 days indicating that they are disabled and are safe to cleanup
        alerts = (
            session.query(DbDynamicAlert)
            .filter(
                or_(
                    DbDynamicAlert.last_queued_at < date_threshold,
                    and_(
                        DbDynamicAlert.last_queued_at.is_(None),
                        DbDynamicAlert.created_at < date_threshold,
                    ),
                )
            )
            .all()
        )

        deleted_count = len(alerts)

        for alert in alerts:
            session.delete(alert)

        session.commit()
        logger.info("Deleted disabled alerts", extra={"count": deleted_count})


@celery_app.task
@sentry_sdk.trace
def cleanup_old_timeseries_and_prophet_history():
    sentry_sdk.set_tag(AnomalyDetectionTags.SEER_FUNCTIONALITY, "anomaly_detection")
    date_threshold = datetime.now() - timedelta(days=90)
    with Session() as session:
        dynamic_alert_stmt = delete(DbDynamicAlertTimeSeriesHistory).where(
            DbDynamicAlertTimeSeriesHistory.timestamp < date_threshold
        )
        dynamic_alert_res = session.execute(dynamic_alert_stmt)

        prophet_alert_stmt = delete(DbProphetAlertTimeSeriesHistory).where(
            DbProphetAlertTimeSeriesHistory.timestamp < date_threshold
        )
        prophet_alert_res = session.execute(prophet_alert_stmt)

        session.commit()
        logger.info(
            "Deleted timeseries history records older than 90 days",
            extra={
                "dynamic alert count": dynamic_alert_res.rowcount,
                "prophet alert count": prophet_alert_res.rowcount,
            },
        )
