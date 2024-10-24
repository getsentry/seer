import abc
import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import sentry_sdk
import stumpy  # type: ignore # mypy throws "missing library stubs"
from pydantic import BaseModel
from sqlalchemy import delete

from seer.anomaly_detection.models import (
    DynamicAlert,
    MPTimeSeries,
    MPTimeSeriesAnomalies,
    TimeSeriesAnomalies,
)
from seer.anomaly_detection.models.cleanup import CleanupConfig
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.db import DbDynamicAlert, DbDynamicAlertTimeSeries, Session, TaskStatus
from seer.exceptions import ClientError

logger = logging.getLogger(__name__)


class AlertDataAccessor(BaseModel, abc.ABC):
    @abc.abstractmethod
    def query(self, external_alert_id: int) -> DynamicAlert | None:
        return NotImplemented

    @abc.abstractmethod
    def save_alert(
        self,
        organization_id: int,
        project_id: int,
        external_alert_id: int,
        config: AnomalyDetectionConfig,
        timeseries: List[TimeSeriesPoint],
        anomalies: TimeSeriesAnomalies,
        anomaly_algo_data: dict,
        data_purge_flag: str,
    ):
        return NotImplemented

    @abc.abstractmethod
    def save_timepoint(
        self,
        external_alert_id: int,
        timepoint: TimeSeriesPoint,
        anomaly: TimeSeriesAnomalies,
        anomaly_algo_data: Optional[dict],
    ):
        return NotImplemented

    @abc.abstractmethod
    def delete_alert_data(self, external_alert_id: int):
        return NotImplemented

    @abc.abstractmethod
    def queue_data_purge_flag(self, external_alert_id: int):
        return NotImplemented

    @abc.abstractmethod
    def can_queue_cleanup_task(self, external_alert_id: int):
        return NotImplemented

    @abc.abstractmethod
    def reset_cleanup_task(self, external_alert_id: int):
        return NotImplemented


class DbAlertDataAccessor(AlertDataAccessor):

    @sentry_sdk.trace
    def _hydrate_alert(self, db_alert: DbDynamicAlert) -> DynamicAlert:
        rand_offset = random.randint(0, 24)
        timestamp_threshold = (datetime.now() - timedelta(days=28, hours=rand_offset)).timestamp()
        num_old_points = 0
        window_size = db_alert.anomaly_algo_data.get("window_size")
        flags = []
        scores = []
        mp = []
        ts = []
        values = []
        for point in db_alert.timeseries:
            ts.append(point.timestamp.timestamp)
            values.append(point.value)
            flags.append(point.anomaly_type)
            scores.append(point.anomaly_score)
            if point.anomaly_algo_data is not None:
                dist, idx, l_idx, r_idx = MPTimeSeriesAnomalies.extract_algo_data(
                    point.anomaly_algo_data
                )
                mp.append([dist, idx, l_idx, r_idx])
            if point.timestamp.timestamp() < timestamp_threshold:
                num_old_points += 1

        anomalies = MPTimeSeriesAnomalies(
            flags=flags,
            scores=scores,
            matrix_profile=stumpy.mparray.mparray(
                mp,
                k=1,
                m=window_size,
                excl_zone_denom=stumpy.config.STUMPY_EXCL_ZONE_DENOM,
            ),
            window_size=window_size,
            thresholds=[],  # Note: thresholds are not stored in the database. They are computed on the fly.
        )
        return DynamicAlert(
            organization_id=db_alert.organization_id,
            project_id=db_alert.project_id,
            external_alert_id=db_alert.external_alert_id,
            config=AnomalyDetectionConfig.model_validate(db_alert.config),
            timeseries=MPTimeSeries(
                timestamps=np.array(ts),
                values=np.array(values),
            ),
            anomalies=anomalies,
            cleanup_config=CleanupConfig(
                num_old_points=num_old_points,
                timestamp_threshold=timestamp_threshold,
                num_acceptable_points=24 * 4 * 2,  # Num alerts for 2 days
            ),
        )

    @sentry_sdk.trace
    def query(self, external_alert_id: int) -> DynamicAlert | None:
        with Session() as session:
            alert_info = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )
            if alert_info is None:
                logger.error(
                    "alert_not_found",
                    extra={
                        "external_alert_id": external_alert_id,
                    },
                )
                return None
            return self._hydrate_alert(alert_info)

    @sentry_sdk.trace
    def save_alert(
        self,
        organization_id: int,
        project_id: int,
        external_alert_id: int,
        config: AnomalyDetectionConfig,
        timeseries: List[TimeSeriesPoint],
        anomalies: TimeSeriesAnomalies,
        anomaly_algo_data: dict,
        data_purge_flag: str,
    ):
        with Session() as session:
            existing_records = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=external_alert_id).count()
            )

            if existing_records > 0:
                # This logic assumes that alert id is unique across organizations and projects.
                # If this assumption changes then alerts can get randomly overwritten.
                logger.info(
                    "overwriting_existing_alert",
                    extra={
                        "organization_id": organization_id,
                        "project_id": project_id,
                        "external_alert_id": external_alert_id,
                    },
                )
                delete_q = delete(DbDynamicAlert).where(
                    DbDynamicAlert.external_alert_id == external_alert_id
                )
                session.execute(delete_q)
            algo_data = anomalies.get_anomaly_algo_data(len(timeseries))
            new_record = DbDynamicAlert(
                organization_id=organization_id,
                project_id=project_id,
                external_alert_id=external_alert_id,
                config=config.model_dump(),
                timeseries=[
                    DbDynamicAlertTimeSeries(
                        timestamp=datetime.fromtimestamp(point.timestamp),
                        value=point.value,
                        anomaly_type=anomalies.flags[i],
                        anomaly_score=anomalies.scores[i],
                        anomaly_algo_data=algo_data[i],
                    )
                    for i, point in enumerate(timeseries)
                ],
                anomaly_algo_data=anomaly_algo_data,
                data_purge_flag=data_purge_flag,
            )
            session.add(new_record)
            session.commit()

    @sentry_sdk.trace
    def save_timepoint(
        self,
        external_alert_id: int,
        timepoint: TimeSeriesPoint,
        anomaly: TimeSeriesAnomalies,
        anomaly_algo_data: Optional[dict],
    ):
        with Session() as session:
            existing = (
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=external_alert_id)
                .one_or_none()
            )
            if existing is None:
                raise ClientError(f"Alert with id {external_alert_id} not found")

            new_record = DbDynamicAlertTimeSeries(
                dynamic_alert_id=existing.id,
                timestamp=datetime.fromtimestamp(timepoint.timestamp),
                value=timepoint.value,
                anomaly_type=anomaly.flags[0],
                anomaly_score=anomaly.scores[0],
                anomaly_algo_data=anomaly_algo_data,
            )
            session.add(new_record)
            session.commit()

    @sentry_sdk.trace
    def delete_alert_data(self, external_alert_id: int):
        with Session() as session:
            existing = (
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=external_alert_id)
                .one_or_none()
            )
            if existing is None:
                raise ClientError(f"Alert with id {external_alert_id} not found")
            session.delete(existing)
            session.commit()

    @sentry_sdk.trace
    def queue_data_purge_flag(self, alert_id: int):
        # Set flag to queued and time when queued
        with Session() as session:
            dynamic_alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == alert_id)
                .one_or_none()
            )

            if not dynamic_alert:
                raise ClientError(f"Alert with id {alert_id} not found")

            dynamic_alert.last_queued_at = datetime.now()
            dynamic_alert.data_purge_flag = TaskStatus.QUEUED
            session.commit()

    @sentry_sdk.trace
    def can_queue_cleanup_task(self, alert_id: int) -> bool:
        """
        Checks if cleanup task can be queued based on current flag or previous time of queueing
        """
        with Session() as session:
            dynamic_alert = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=alert_id).one_or_none()
            )

            if not dynamic_alert:
                raise ClientError(f"Alert with id {alert_id} not found")

            queued_at_threshold = datetime.now() - timedelta(hours=12)

            return (
                dynamic_alert.data_purge_flag == TaskStatus.NOT_QUEUED
                or dynamic_alert.last_queued_at is not None
                and dynamic_alert.last_queued_at < queued_at_threshold
            )

    @sentry_sdk.trace
    def reset_cleanup_task(self, alert_id: int) -> None:
        with Session() as session:
            dynamic_alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == alert_id)
                .one_or_none()
            )

            if not dynamic_alert:
                raise ClientError(f"Alert with id {alert_id} not found")

            dynamic_alert.last_queued_at = None
            dynamic_alert.data_purge_flag = TaskStatus.NOT_QUEUED
            session.commit()
