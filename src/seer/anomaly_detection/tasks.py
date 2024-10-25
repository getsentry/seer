import logging

import numpy as np
import sentry_sdk

from celery_app.app import celery_app
from seer.anomaly_detection.detectors.anomaly_detectors import MPBatchAnomalyDetector
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.anomaly_detection.models.timeseries import TimeSeries
from seer.db import DbDynamicAlert, Session, TaskStatus

logger = logging.getLogger(__name__)


@celery_app.task
@sentry_sdk.trace
def cleanup_timeseries(alert_id: int, date_threshold: float):
    transaction = sentry_sdk.get_current_scope().transaction

    if transaction is not None:
        transaction.set_tag("alert_id", alert_id)

    logger.info("Deleting timeseries points over 28 days old and updating matrix profiles")
    toggle_data_purge_flag(alert_id)

    with Session() as session:
        alert = (
            session.query(DbDynamicAlert)
            .filter(DbDynamicAlert.external_alert_id == alert_id)
            .one_or_none()
        )

        if alert is None:
            raise ValueError(f"Alert with id {alert_id} not found")
        if len(alert.timeseries) == 0:
            logger.warn(f"Alert with id {alert_id} has no timeseries")
        else:
            config = AnomalyDetectionConfig(
                time_period=alert.config["time_period"],
                sensitivity=alert.config["sensitivity"],
                direction=alert.config["direction"],
                expected_seasonality=alert.config["expected_seasonality"],
            )
            deleted_timeseries_points = delete_old_timeseries_points(alert, date_threshold)
            updated_timeseries_points = update_matrix_profiles(alert, config)
            session.commit()
            logger.info(f"Deleted {deleted_timeseries_points} timeseries points")
            logger.info(
                f"Updated matrix profiles for {updated_timeseries_points} points in alertd id {alert_id}"
            )

    toggle_data_purge_flag(alert_id)


def delete_old_timeseries_points(alert: DbDynamicAlert, date_threshold: float):
    deleted_count = 0
    to_remove = []
    for ts in alert.timeseries:
        if ts.timestamp.timestamp() < date_threshold:
            to_remove.append(ts)
    for ts in to_remove:
        alert.timeseries.remove(ts)
        deleted_count += 1
    return deleted_count


def update_matrix_profiles(alert: DbDynamicAlert, anomaly_detection_config: AnomalyDetectionConfig):

    timeseries = TimeSeries(
        timestamps=np.array([timestep.timestamp for timestep in alert.timeseries]),
        values=np.array([timestep.value for timestep in alert.timeseries]),
    )

    anomalies = MPBatchAnomalyDetector()._compute_matrix_profile(
        timeseries=timeseries, config=anomaly_detection_config
    )
    algo_data_map = dict(
        zip(timeseries.timestamps, anomalies.get_anomaly_algo_data(len(timeseries.timestamps)))
    )
    updateed_timeseries_points = 0
    for timestep in alert.timeseries:
        timestep.anomaly_algo_data = algo_data_map[timestep.timestamp]
        updateed_timeseries_points += 1
    alert.anomaly_algo_data = {"window_size": anomalies.window_size}
    return updateed_timeseries_points


def toggle_data_purge_flag(alert_id: int):

    with Session() as session:
        alert = (
            session.query(DbDynamicAlert)
            .filter(DbDynamicAlert.external_alert_id == alert_id)
            .one_or_none()
        )

        if alert is None:
            raise ValueError(f"Alert with id {alert_id} not found")
        new_flag = (
            TaskStatus.PROCESSING
            if alert.data_purge_flag == TaskStatus.QUEUED
            else TaskStatus.NOT_QUEUED
        )
        alert.data_purge_flag = new_flag
        session.commit()
