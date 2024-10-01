import abc
import datetime
import logging
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
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.db import DbDynamicAlert, DbDynamicAlertTimeSeries, Session
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


class DbAlertDataAccessor(AlertDataAccessor):

    @sentry_sdk.trace
    def _hydrate_alert(self, db_alert: DbDynamicAlert) -> DynamicAlert:
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
                        timestamp=datetime.datetime.fromtimestamp(point.timestamp),
                        value=point.value,
                        anomaly_type=anomalies.flags[i],
                        anomaly_score=anomalies.scores[i],
                        anomaly_algo_data=algo_data[i],
                    )
                    for i, point in enumerate(timeseries)
                ],
                anomaly_algo_data=anomaly_algo_data,
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
                timestamp=datetime.datetime.fromtimestamp(timepoint.timestamp),
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
