import logging
from datetime import datetime, timedelta
from operator import and_, or_
from typing import List

import numpy as np
import sentry_sdk

from celery_app.app import celery_app
from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.detectors.anomaly_detectors import MPBatchAnomalyDetector
from seer.anomaly_detection.models import AlgoConfig, TimeSeries
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.db import (
    DbDynamicAlert,
    DbDynamicAlertTimeSeries,
    DbDynamicAlertTimeSeriesHistory,
    Session,
    TaskStatus,
)
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


@celery_app.task
@sentry_sdk.trace
def cleanup_timeseries(alert_id: int, date_threshold: float):
    span = sentry_sdk.get_current_span()

    if span is not None:
        span.set_tag("alert_id", alert_id)

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
            if len(alert.timeseries) > 0:
                updated_timeseries_points = update_matrix_profiles(alert, config)
            else:
                # Reset the window size to 0 if there are no timeseries points left
                alert.anomaly_algo_data = {"window_size": 0}
                logger.warn(f"Alert with id {alert_id} has empty timeseries data after pruning")
                updated_timeseries_points = 0
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

    # Save history records before removing
    save_timeseries_history(alert, to_remove)

    for ts in to_remove:
        alert.timeseries.remove(ts)
        deleted_count += 1
    return deleted_count


def save_timeseries_history(alert: DbDynamicAlert, timeseries: List[DbDynamicAlertTimeSeries]):
    with Session() as session:
        for ts in timeseries:
            history_record = DbDynamicAlertTimeSeriesHistory(
                alert_id=alert.external_alert_id,
                timestamp=ts.timestamp,
                anomaly_type=ts.anomaly_type,
                value=ts.value,
                saved_at=datetime.datetime.now(datetime.UTC),
            )
            session.add(history_record)
        session.commit()


@inject
def update_matrix_profiles(
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
    anomalies_fixed = MPBatchAnomalyDetector()._compute_matrix_profile(
        timeseries=timeseries,
        ad_config=anomaly_detection_config,
        algo_config=algo_config,
        window_size=algo_config.mp_fixed_window_size,
    )
    anomalies = DbAlertDataAccessor().combine_anomalies(
        anomalies_suss, anomalies_fixed, [True] * len(timeseries.timestamps)
    )

    algo_data_map = dict(
        zip(timeseries.timestamps, anomalies.get_anomaly_algo_data(len(timeseries.timestamps)))
    )
    updateed_timeseries_points = 0
    for timestep in alert.timeseries:
        timestep.anomaly_algo_data = algo_data_map[timestep.timestamp.timestamp()]
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


@celery_app.task
@sentry_sdk.trace
def cleanup_disabled_alerts():

    date_threshold = datetime.now() - timedelta(days=28)

    logger.info(
        f"Cleaning up timeseries data for alerts that have been inactive (detection has not been run) since {date_threshold}"
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
        logger.info(f"Deleted {deleted_count} alerts")


@celery_app.task
@sentry_sdk.trace
def cleanup_old_timeseries_history():
    date_threshold = datetime.now() - timedelta(days=90)
    with Session() as session:
        deleted_count = (
            session.query(DbDynamicAlertTimeSeriesHistory)
            .filter(DbDynamicAlertTimeSeriesHistory.saved_at < date_threshold)
            .delete()
        )
        session.commit()
        logger.info(f"Deleted {deleted_count} timeseries history records older than 90 days")
