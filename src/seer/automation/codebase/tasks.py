import datetime
import logging
import time
from datetime import timedelta

from pydantic import BaseModel

from celery_app.app import celery_app
from seer.automation.codebase.repo_client import RepoClient, RepoClientType
from seer.automation.codebase.repo_manager import RepoManager
from seer.automation.models import RepoDefinition
from seer.configuration import AppConfig
from seer.db import (
    DbSeerBackfillJob,
    DbSeerBackfillState,
    DbSeerProjectPreference,
    DbSeerRepoArchive,
    Session,
)
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class BackfillJob(BaseModel):
    organization_id: int
    repo_definition: RepoDefinition
    backfill_job_id: int | None = None


class BackfillJobError(RuntimeError):
    pass


@celery_app.task(
    soft_time_limit=timedelta(minutes=1).total_seconds(),
    time_limit=timedelta(minutes=1, seconds=10).total_seconds(),
)
def collect_all_repos_for_backfill():
    logger.info("Collecting repos for backfill")

    # TODO: Remove this once we have a real backfill cursor
    backfill_project_ids = [1, 300688, 1320254, 5613870, 4507936016302080]

    with Session() as session:
        backfill_state = (
            session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
        )

        # TODO: Implement backfill cursor
        # ...

        # start of hack #
        # This logic below with the task_taken_key is basically because of how our celerybeat setup is.
        # Every deployment of celerybeat will each schedule a task, so this is a hack to ensure that only one
        # deployment runs the task at a time.
        task_taken_key = backfill_state.task_taken_key if backfill_state else None

        # current timestamp, rounded to the nearest 30 minutes
        current_cron_iteration = str(round(time.time() / 1800))

        print(f"current_cron_iteration: {current_cron_iteration}")
        print(f"task_taken_key: {task_taken_key}, {task_taken_key == current_cron_iteration}")

        # if task_taken_key == current_cron_iteration:
        #     # don't do anything, we've already run this cron iteration
        #     logger.info("Already ran this cron iteration, skipping")
        #     return

        # update the task taken key
        if backfill_state:
            backfill_state.task_taken_key = current_cron_iteration
        else:
            session.add(
                DbSeerBackfillState(id=1, backfill_cursor=0, task_taken_key=current_cron_iteration)
            )

        session.commit()

        # end of hack #

        project_preferences = (
            session.query(DbSeerProjectPreference)
            .filter(DbSeerProjectPreference.project_id.in_(backfill_project_ids))
            .order_by(DbSeerProjectPreference.project_id)
            # TODO: for backfill cursor .limit(MAX_QUERIED_PROJECT_IDS)
            .all()
        )

    if len(project_preferences) == 0:
        logger.info("No project preferences to backfill")
        return

    logger.info(f"Found {len(project_preferences)} project preferences to backfill")

    backfill_jobs: list[BackfillJob] = []

    for project_preference in project_preferences:
        for repo in project_preference.repositories:
            repo_definition = RepoDefinition.model_validate(repo)

            with Session() as session:
                existing_archive = (
                    session.query(DbSeerRepoArchive)
                    .filter(
                        DbSeerRepoArchive.organization_id == project_preference.organization_id,
                        DbSeerRepoArchive.bucket_name == RepoManager.get_bucket_name(),
                        DbSeerRepoArchive.blob_path
                        == RepoManager.make_blob_name(
                            project_preference.organization_id,
                            repo_definition.provider,
                            repo_definition.owner,
                            repo_definition.name,
                            repo_definition.external_id,
                        ),
                    )
                    .first()
                )

            if not existing_archive:
                backfill_jobs.append(
                    BackfillJob(
                        organization_id=project_preference.organization_id,
                        repo_definition=repo_definition,
                    )
                )
            else:
                logger.info(
                    f"Repo {repo_definition.full_name} for org {project_preference.organization_id} already exists in archive with id {existing_archive.id}, skipping."
                )

    logger.info(f"Collected {len(backfill_jobs)} repos for backfill")

    with Session() as session:
        # TODO: Implement backfill cursor
        # session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).update(
        #     {DbSeerBackfillState.backfill_cursor: new_backfill_cursor}
        # )

        # create backfill jobs
        for backfill_job in backfill_jobs:
            db_backfill_job = DbSeerBackfillJob(
                organization_id=backfill_job.organization_id,
                repo_provider=backfill_job.repo_definition.provider,
                repo_external_id=backfill_job.repo_definition.external_id,
            )

            session.add(db_backfill_job)
            session.flush()

            backfill_job.backfill_job_id = db_backfill_job.id

        session.commit()

    for backfill_job in backfill_jobs:
        run_backfill.apply_async(args=(backfill_job.model_dump(),))


@inject
def run_backfill_task(app_config: AppConfig = injected):
    run_backfill.apply_async(queue=app_config.CELERY_WORKER_QUEUE)


@celery_app.task(
    soft_time_limit=timedelta(minutes=15).total_seconds(),
    time_limit=timedelta(minutes=17).total_seconds(),
)
def run_backfill(backfill_job_dict: dict):
    backfill_job = BackfillJob.model_validate(backfill_job_dict)

    if not backfill_job.backfill_job_id:
        raise BackfillJobError("backfill_job_id is required")

    logger.info(
        f"Running backfill job {backfill_job.backfill_job_id} for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id}"
    )

    with Session() as session:
        db_backfill_job = (
            session.query(DbSeerBackfillJob)
            .filter(DbSeerBackfillJob.id == backfill_job.backfill_job_id)
            .first()
        )

        if not db_backfill_job:
            raise BackfillJobError("backfill job not found")

        if db_backfill_job.started_at:
            raise BackfillJobError("backfill job already started")

        if db_backfill_job.completed_at:
            raise BackfillJobError("backfill job already completed")

        if db_backfill_job.failed_at:
            raise BackfillJobError("backfill job already failed")

        db_backfill_job.started_at = datetime.datetime.now(datetime.UTC)

        session.commit()

    try:
        repo_client = RepoClient.from_repo_definition(
            backfill_job.repo_definition, RepoClientType.READ
        )
        repo_manager = RepoManager(
            repo_client, organization_id=backfill_job.organization_id, force_gcs=True
        )

        repo_manager.initialize_archive_for_backfill()

        logger.info("Backfill job done.")

        with Session() as session:
            session.query(DbSeerBackfillJob).filter(
                DbSeerBackfillJob.id == backfill_job.backfill_job_id
            ).update(
                {
                    DbSeerBackfillJob.completed_at: datetime.datetime.now(datetime.UTC),
                }
            )
            session.commit()

        run_test_download_and_verify_backfill.apply_async(args=(backfill_job_dict,))
    except Exception:
        logger.exception(
            "Failed to run backfill job", extra={"backfill_job_id": backfill_job.backfill_job_id}
        )

        with Session() as session:
            session.query(DbSeerBackfillJob).filter(
                DbSeerBackfillJob.id == backfill_job.backfill_job_id
            ).update(
                {
                    DbSeerBackfillJob.failed_at: datetime.datetime.now(datetime.UTC),
                }
            )
            session.commit()
            raise
    finally:
        repo_manager.cleanup()


@celery_app.task(
    soft_time_limit=timedelta(minutes=10).total_seconds(),
    time_limit=timedelta(minutes=12).total_seconds(),
)
def run_test_download_and_verify_backfill(backfill_job_dict: dict):
    backfill_job = BackfillJob.model_validate(backfill_job_dict)

    logger.info(
        f"Running test download and verify backfill for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id}"
    )

    with Session() as session:
        db_backfill_job = (
            session.query(DbSeerBackfillJob)
            .filter(DbSeerBackfillJob.id == backfill_job.backfill_job_id)
            .first()
        )

        if not db_backfill_job:
            raise BackfillJobError("backfill job not found")

        if db_backfill_job.verified_at:
            raise BackfillJobError("backfill job already verified")

    repo_client = RepoClient.from_repo_definition(backfill_job.repo_definition, RepoClientType.READ)
    repo_manager = RepoManager(
        repo_client, organization_id=backfill_job.organization_id, force_gcs=True
    )

    repo_manager.initialize()

    logger.info("Backfill verification job done.")

    with Session() as session:
        session.query(DbSeerBackfillJob).filter(
            DbSeerBackfillJob.id == backfill_job.backfill_job_id
        ).update(
            {
                DbSeerBackfillJob.verified_at: datetime.datetime.now(datetime.UTC),
            }
        )
        session.commit()

    repo_manager.cleanup()
