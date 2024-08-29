import abc
import datetime
import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel
from sqlalchemy import delete

from seer.anomaly_detection.models import DynamicAlert, MPTimeSeries, TimeSeriesAnomalies
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.db import DbDynamicAlert, DbDynamicAlertTimeSeries, Session

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


class DbAlertDataAccessor(AlertDataAccessor):
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

            timeseries: MPTimeSeries = MPTimeSeries(
                timestamps=np.empty([len(alert_info.timeseries)]),
                values=np.empty([len(alert_info.timeseries)]),
                window_size=alert_info.anomaly_algo_data.get("window_size"),
            )

            for i, point in enumerate(alert_info.timeseries):
                np.put(timeseries.timestamps, i, point.timestamp.timestamp())
                np.put(timeseries.values, i, point.value)

            return DynamicAlert(
                organization_id=alert_info.organization_id,
                project_id=alert_info.project_id,
                external_alert_id=external_alert_id,
                config=AnomalyDetectionConfig.model_validate(alert_info.config),
                timeseries=timeseries,
            )

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
            algo_data = anomalies.get_anomaly_algo_data()
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
                raise Exception(f"Alert with id {external_alert_id} not found")

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
