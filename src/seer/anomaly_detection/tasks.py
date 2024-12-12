import logging
from datetime import datetime, timedelta
from operator import and_, or_

import numpy as np
import sentry_sdk

from celery_app.app import celery_app
from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.detectors.anomaly_detectors import MPBatchAnomalyDetector
from seer.anomaly_detection.models import AlgoConfig, TimeSeries
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.db import DbDynamicAlert, Session, TaskStatus
from seer.dependency_injection import inject, injected
from seer.anomaly_detection.utils import chunks, with_retry, safe_commit

logger = logging.getLogger(__name__)

CHUNK_SIZE = 100  # Process data in chunks of 100 records

@celery_app.task
@with_retry(max_retries=3)
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
            
            # Delete old points in chunks
            deleted_timeseries_points = 0
            for chunk in chunks(list(alert.timeseries), CHUNK_SIZE):
                deleted_count = delete_old_timeseries_points(chunk, date_threshold)
                deleted_timeseries_points += deleted_count
                safe_commit(session)
            
            if len(alert.timeseries) > 0:
                total_updated = 0
                for chunk in chunks(list(alert.timeseries), CHUNK_SIZE):
                    updated_count = update_matrix_profiles_chunk(alert, chunk, config)
                    total_updated += updated_count
                    safe_commit(session)
            else:
                alert.anomaly_algo_data = {"window_size": 0}

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

def update_matrix_profiles_chunk(
        alert: DbDynamicAlert,
        chunk: List[DbDynamicAlert],
        anomaly_detection_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig = injected,
    ):
    """Process a chunk of timeseries data for matrix profile updates."""
    try:
        timeseries = TimeSeries(
            timestamps=np.array([ts.timestamp.timestamp() for ts in chunk]),
            values=np.array([ts.value for ts in chunk])
        )

        detector = MPBatchAnomalyDetector()
        anomalies_suss = detector._compute_matrix_profile(
            timeseries=timeseries, ad_config=anomaly_detection_config, algo_config=algo_config
        )
        anomalies_fixed = detector._compute_matrix_profile(
            timeseries=timeseries,
            ad_config=anomaly_detection_config,
            algo_config=algo_config,
            window_size=algo_config.mp_fixed_window_size,
        )
        
        anomalies = DbAlertDataAccessor().combine_anomalies(
            anomalies_suss, anomalies_fixed, [True] * len(timeseries.timestamps)
        )

        algo_data = anomalies.get_anomaly_algo_data(len(timeseries.timestamps))
        for ts, data in zip(chunk, algo_data):
            ts.anomaly_algo_data = data
            
        alert.anomaly_algo_data = {"window_size": anomalies.window_size}
        return len(chunk)
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        raise


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
