import datetime
import logging

import sentry_sdk
import sqlalchemy.sql as sql

from celery_app.app import celery_app
from seer.db import DbIssueSummary, DbRunState, Session

logger = logging.getLogger(__name__)


@celery_app.task(time_limit=30)
def delete_data_for_ttl():
    logger.info("Deleting old automation runs and issue summaries for 30 day time-to-live")
    before = datetime.datetime.now() - datetime.timedelta(days=30)  # over 30 days old
    deleted_run_count = delete_all_runs_before(before)
    deleted_summary_count = delete_all_summaries_before(before)
    logger.info(f"Deleted {deleted_run_count} runs")
    logger.info(f"Deleted {deleted_summary_count} summaries")


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


def delete_all_summaries_before(before: datetime.datetime, batch_size=1000):
    deleted_count = 0
    while True:
        with Session() as session:
            subquery = (
                session.query(DbIssueSummary.group_id)
                .filter(DbIssueSummary.created_at < before)
                .limit(batch_size)
                .subquery()
            )
            count = (
                session.query(DbIssueSummary)
                .filter(sql.exists().where(DbIssueSummary.group_id == subquery.c.group_id))
                .delete()
            )
            session.commit()

            deleted_count += count
            if count == 0:
                break
            sentry_sdk.metrics.incr(
                key="issue_summary_TTL_deletion",
                value=count,
            )

    return deleted_count
