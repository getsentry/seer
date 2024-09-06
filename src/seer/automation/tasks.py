import datetime
import logging

import sentry_sdk
import sqlalchemy.sql as sql

from celery_app.app import celery_app
from seer.db import DbRunState, Session

logger = logging.getLogger(__name__)


@celery_app.task(time_limit=30)
def delete_old_automation_runs():
    logger.info("Deleting old Autofix & Unittest generation runs for 90 day time-to-live")
    before = datetime.datetime.now() - datetime.timedelta(days=90)  # over 90 days old
    deleted_count = delete_all_runs_before(before)
    print(deleted_count)
    logger.info(f"Deleted {deleted_count} runs")


def delete_all_runs_before(before: datetime.datetime, batch_size=1000):
    deleted_count = 0
    while True:
        with Session() as session:
            subquery = (
                session.query(DbRunState.id)
                .filter(DbRunState.last_triggered_at < before)
                .limit(batch_size)
                .subquery()
            )
            count = (
                session.query(DbRunState)
                .filter(sql.exists().where(DbRunState.id == subquery.c.id))
                .delete()
            )
            session.commit()

            deleted_count += count
            if count == 0:
                break
            sentry_sdk.metrics.incr(
                key="autofix_state_TTL_deletion",
                value=count,
            )

    return deleted_count
