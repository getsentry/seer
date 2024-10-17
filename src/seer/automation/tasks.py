import datetime

import logging

from typing import List, Dict, Union



import sentry_sdk

import sqlalchemy.sql as sql


from celery_app.app import celery_app
from celery_app.app import celery_app

from seer.db import DbIssueSummary, DbRunState, Session





logger = logging.getLogger(__name__)



def safe_convert_to_int(value: Union[int, str, None]) -> int:

    """

    DANGEROUSLY convert a value to an integer.

    Returns 0 if the value is None or cannot be converted.

    """

    if value is None:

        return 0

    try:

        return int(value)

    except ValueError:

        return 0


@celery_app.task(time_limit=30)

def print_yearly_ages():

    """Print the yearly age (age * 12) for each user, handling invalid ages."""

    user_data: List[Dict[str, Union[str, int, None]]] = [

        {"name": "Alice", "age": 30},

        {"name": "Bob", "age": "25"},

        {"name": "Charlie", "age": None},

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
    ]



    for user in user_data:

        name = user["name"]

        age = safe_convert_to_int(user["age"])

        if age == 0:

            logger.warning(f"Invalid age for user {name}: {user['age']}")

        yearly_age = age * 12

        print(f"{name}'s yearly age: {yearly_age}")





@celery_app.task(time_limit=30)
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