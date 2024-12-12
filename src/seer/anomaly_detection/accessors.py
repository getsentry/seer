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
    AlgoConfig,
    DynamicAlert,
    MPTimeSeries,
    MPTimeSeriesAnomalies,
    MPTimeSeriesAnomaliesSingleWindow,
    TimeSeriesAnomalies,
)
from seer.anomaly_detection.models.cleanup import CleanupConfig
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.db import DbDynamicAlert, DbDynamicAlertTimeSeries, Session, TaskStatus
from seer.dependency_injection import inject, injected
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

    @abc.abstractmethod
    def combine_anomalies(
        self,
        anomalies_suss: MPTimeSeriesAnomaliesSingleWindow,
        anomalies_fixed: MPTimeSeriesAnomaliesSingleWindow | None,
        use_suss: list[bool],
    ):
        return NotImplemented


class DbAlertDataAccessor(AlertDataAccessor):

    @inject
    @sentry_sdk.trace
    def _hydrate_alert(
        self,
        db_alert: DbDynamicAlert,
        algo_config: AlgoConfig = injected,
    ) -> DynamicAlert:
        rand_offset = random.randint(0, 24)
        timestamp_threshold = (datetime.now() - timedelta(days=28, hours=rand_offset)).timestamp()
        num_old_points = 0
        window_size = db_alert.anomaly_algo_data.get("window_size")
        flags = []
        scores = []
        mp_suss = []
        mp_fixed = []
        ts = []
        values = []
        original_flags = []
        use_suss = []

        timeseries = db_alert.timeseries

        # If the timeseries does not have both matrix profiles, then we only use the suss window
        only_suss = False
        if len(timeseries) > 0 and any(
            point.anomaly_algo_data is not None
            and "mp_suss" not in point.anomaly_algo_data
            and "mp_fixed" not in point.anomaly_algo_data
            for point in timeseries
        ):
            only_suss = True

        for point in timeseries:
            ts.append(point.timestamp.timestamp())
            values.append(point.value)
            flags.append(point.anomaly_type)
            scores.append(point.anomaly_score)

            if point.anomaly_algo_data is not None:
                algo_data = MPTimeSeriesAnomalies.extract_algo_data(point.anomaly_algo_data)

                if "mp_suss" in algo_data and algo_data["mp_suss"]:
                    mp_suss_data = [
                        algo_data["mp_suss"]["dist"],
                        algo_data["mp_suss"]["idx"],
                        algo_data["mp_suss"]["l_idx"],
                        algo_data["mp_suss"]["r_idx"],
                    ]
                    mp_suss.append(mp_suss_data)

                if "mp_fixed" in algo_data and algo_data["mp_fixed"]:
                    mp_fixed_data = [
                        algo_data["mp_fixed"]["dist"],
                        algo_data["mp_fixed"]["idx"],
                        algo_data["mp_fixed"]["l_idx"],
                        algo_data["mp_fixed"]["r_idx"],
                    ]
                    mp_fixed.append(mp_fixed_data)
                use_suss.append(algo_data["use_suss"])
                original_flags.append(algo_data["original_flag"])

            if point.timestamp.timestamp() < timestamp_threshold:
                num_old_points += 1
        # Default value is "none" for original flags
        if len(original_flags) < len(ts):
            original_flags = ["none"] * (len(ts) - len(original_flags)) + original_flags

        # Default value is True for use_suss
        if len(use_suss) < len(ts):
            use_suss = [True] * (len(ts) - len(use_suss)) + use_suss

        anomalies = MPTimeSeriesAnomalies(
            flags=flags,
            scores=scores,
            matrix_profile_suss=stumpy.mparray.mparray(
                mp_suss,
                k=1,
                m=window_size,
                excl_zone_denom=stumpy.config.STUMPY_EXCL_ZONE_DENOM,
            ),
            matrix_profile_fixed=stumpy.mparray.mparray(
                mp_fixed,
                k=1,
                m=algo_config.mp_fixed_window_size,
                excl_zone_denom=stumpy.config.STUMPY_EXCL_ZONE_DENOM,
            ),
            window_size=window_size,
            thresholds=[],  # Note: thresholds are not stored in the database. They are computed on the fly.
            original_flags=original_flags,
            use_suss=use_suss,
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
            only_suss=only_suss,
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

    def combine_anomalies(
        self,
        anomalies_suss: MPTimeSeriesAnomaliesSingleWindow,
        anomalies_fixed: MPTimeSeriesAnomaliesSingleWindow | None,
        use_suss: list[bool],
    ) -> MPTimeSeriesAnomalies:
        """
        Combines anomalies detected using SuSS and fixed window approaches into a single MPTimeSeriesAnomalies object.
        For each point, uses either the SuSS or fixed window anomaly based on the use_suss flag.

        Parameters:
        anomalies_suss: MPTimeSeriesAnomalies
            Anomalies detected using the SuSS window
        anomalies_fixed: MPTimeSeriesAnomalies
            Anomalies detected using the fixed window
        use_suss: list[bool]
            Flags indicating whether to use the SuSS or fixed window for each point

        Returns:
        MPTimeSeriesAnomalies
            Combined anomalies object containing flags, scores and metadata from both approaches
        """
        combined_flags = anomalies_suss.flags
        combined_scores = anomalies_suss.scores
        combined_thresholds = anomalies_suss.thresholds
        combined_original_flags = anomalies_suss.original_flags
        if anomalies_fixed is not None:
            for i in range(len(anomalies_suss.flags)):
                if use_suss[i]:
                    combined_flags[i] = anomalies_suss.flags[i]
                    combined_scores[i] = anomalies_suss.scores[i]
                    combined_original_flags[i] = anomalies_suss.original_flags[i]
                    if i < len(anomalies_suss.thresholds):
                        combined_thresholds[i] = anomalies_suss.thresholds[i]
                else:
                    combined_flags[i] = anomalies_fixed.flags[i]
                    combined_scores[i] = anomalies_fixed.scores[i]
                    combined_original_flags[i] = anomalies_fixed.original_flags[i]
                    if i < len(anomalies_fixed.thresholds):
                        combined_thresholds[i] = anomalies_fixed.thresholds[i]

        return MPTimeSeriesAnomalies(
            flags=combined_flags,
            scores=combined_scores,
            thresholds=combined_thresholds,
            matrix_profile_suss=anomalies_suss.matrix_profile,
            matrix_profile_fixed=(
                anomalies_fixed.matrix_profile if anomalies_fixed is not None else np.array([])
            ),
            window_size=anomalies_suss.window_size,
            original_flags=combined_original_flags,
            use_suss=use_suss,
        )
