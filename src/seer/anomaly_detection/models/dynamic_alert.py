import datetime
import logging
from typing import List

from pydantic import BaseModel
from sqlalchemy import delete

from seer.anomaly_detection.models.external import TimeSeriesPoint
from seer.db import DbDynamicAlert, DbDynamicAlertTimeSeries, Session

logger = logging.getLogger(__name__)


class DynamicAlert(BaseModel):
    organization_id: int
    project_id: int
    external_alert_id: int
    config: dict

    @classmethod
    def from_db(cls, db_repo: DbDynamicAlert) -> "DynamicAlert":
        return cls(
            organization_id=db_repo.organization_id,
            project_id=db_repo.project_id,
            external_alert_id=db_repo.external_alert_id,
            config=db_repo.config,
        )

    def save(self, timeseries: List[TimeSeriesPoint]):
        with Session() as session:
            existing_records = (
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=self.external_alert_id)
                .count()
            )

            if existing_records > 0:
                logger.info(
                    "overwriting_existing_alert",
                    extra={
                        "organization_id": self.organization_id,
                        "project_id": self.project_id,
                        "external_alert_id": self.external_alert_id,
                    },
                )
                delete_q = delete(DbDynamicAlertTimeSeries).where(
                    DbDynamicAlertTimeSeries.external_alert_id == self.external_alert_id
                )
                session.execute(delete_q)
                delete_q = delete(DbDynamicAlert).where(
                    DbDynamicAlert.external_alert_id == self.external_alert_id
                )
                session.execute(delete_q)

            new_record = self.to_db_model()
            for point in timeseries:
                new_record.timeseries.append(
                    DbDynamicAlertTimeSeries(
                        external_alert_id=self.external_alert_id,
                        timestamp=datetime.datetime.fromtimestamp(point.timestamp),
                        value=point.value,
                    )
                )
            session.add(new_record)
            session.commit()

    def to_db_model(self) -> DbDynamicAlert:
        return DbDynamicAlert(
            organization_id=self.organization_id,
            project_id=self.project_id,
            external_alert_id=self.external_alert_id,
            config=self.config,
        )


class DynamicAlertTimeSeries(BaseModel):
    external_alert_id: int
    timestamp: datetime.datetime
    value: float

    def to_db_model(self) -> DbDynamicAlertTimeSeries:
        return DbDynamicAlertTimeSeries(
            external_alert_id=self.external_alert_id,
            timestamp=self.timestamp,
            value=self.value,
        )
