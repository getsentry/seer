import datetime
import logging

import sentry_sdk
import sqlalchemy.sql as sql

from celery_app.app import celery_app
from seer.db import DbIssueSummary, DbRunState, Session
from seer.automation.state import DbState, DbStateRunTypes
from seer.automation.codegen.state import CodegenContinuation
from seer.automation.codegen.models import CodegenStatus

logger = logging.getLogger(__name__)


@celery_app.task(time_limit=30)
def delete_data_for_ttl():
    logger.info("Deleting old automation runs and issue summaries for 90 day time-to-live")
    before = datetime.datetime.now() - datetime.timedelta(days=90)  # over 90 days old
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


def spawn_pr_reaction_checks():
    """
    Regular function that spawns PR reaction check tasks
    """
    with Session() as session:
        db_states = session.query(DbRunState).filter(
            DbRunState.type == DbStateRunTypes.PR_REVIEW,
            DbRunState.group_id.isnot(None)
        ).all()
        state_ids = [state.id for state in db_states]

    for state_id in state_ids:
        check_pr_reactions.delay(state_id)


@celery_app.task(
    rate_limit='100/m',
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=30,
)
def check_pr_reactions(db_state_id: int):
    """
    Task to check reactions for a single PR
    """
    try:
        state = DbState(
            id=db_state_id,
            model=CodegenContinuation,
            type=DbStateRunTypes.PR_REVIEW
        )
        continuation = state.get()
        
        # Skip if not completed
        if continuation.status != CodegenStatus.COMPLETED:
            return

        # Get the PR ID
        with Session() as session:
            db_state = session.query(DbRunState).get(db_state_id)
            if not db_state or not db_state.group_id:
                return
            pr_id = db_state.group_id

        # TODO: Implement GitHub reaction checking logic here
        # You'll handle the GitHub API calls and DB operations

    except Exception as e:
        logger.error(f"Error processing state {db_state_id}: {e}", exc_info=True)
        raise  # Let Celery handle the retry


@celery_app.task
def check_github_reactions():
    """
    Periodic task that spawns individual tasks for checking GitHub reactions on PRs
    """
    with Session() as session:
        db_states = session.query(DbRunState).filter(
            DbRunState.type == DbStateRunTypes.PR_REVIEW,
            DbRunState.group_id.isnot(None)  # Ensure we have a PR ID
        ).all()
        state_ids = [state.id for state in db_states]

    for state_id in state_ids:
        check_pr_reactions.delay(state_id)
