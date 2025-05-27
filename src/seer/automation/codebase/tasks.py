import datetime
import logging
from contextlib import contextmanager
from datetime import timedelta

from pydantic import BaseModel
from sqlalchemy import text

from celery_app.app import celery_app
from seer.automation.codebase.repo_client import RepoClient, RepoClientType
from seer.automation.codebase.repo_manager import RepoManager
from seer.automation.codebase.utils import ensure_timezone_aware
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

MAX_QUERIED_PROJECT_IDS = 250
THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING = 32
MAX_QUERIED_REPO_ARCHIVES_PER_SYNC = 250
MAX_REPO_ARCHIVES_PER_SYNC = 32


@contextmanager
def acquire_backfill_lock(session):
    """
    Acquire a Postgres advisory lock for the backfill task.
    The lock key is a 64-bit integer: we use a simple hash of 'backfill_scheduler' string.
    """
    # This ensures the same lock key is used across restarts
    lock_key = 1234567890123456789

    try:
        # pg_try_advisory_xact_lock returns true if lock acquired, false if not
        # This is a transaction-level lock that auto-releases when transaction ends
        got_lock = session.execute(
            text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": lock_key}
        ).scalar()

        if not got_lock:
            logger.info("Could not acquire backfill lock, another process has it")
            yield False
        else:
            logger.info("Acquired backfill lock")
            yield True

    except Exception:
        logger.exception("Error while managing backfill lock")
        yield False


class BackfillJob(BaseModel):
    organization_id: int
    repo_definition: RepoDefinition
    backfill_job_id: int | None = None
    scaled_time_limit: float = timedelta(minutes=15).total_seconds()


class BackfillJobError(RuntimeError):
    pass


class RepoSyncJob(BaseModel):
    archive_id: int
    scaled_time_limit: float = timedelta(minutes=15).total_seconds()


class RepoSyncJobError(RuntimeError):
    pass


@celery_app.task(
    soft_time_limit=timedelta(minutes=1).total_seconds(),
    time_limit=timedelta(minutes=1, seconds=10).total_seconds(),
)
def collect_all_repos_for_backfill():
    logger.info("Collecting repos for backfill")

    backfill_jobs: list[BackfillJob] = []
    processed_repos = set()  # Tracks (org_id, repo_provider, repo_external_id)

    with Session() as main_session:
        backfill_state = (
            main_session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
        )

        if not backfill_state:
            backfill_state = DbSeerBackfillState(id=1, backfill_cursor=0)
            main_session.add(backfill_state)
            main_session.flush()

        # Use advisory lock instead of task_taken_key
        with acquire_backfill_lock(main_session) as got_lock:
            if not got_lock:
                return

            # If we get here, we have the lock
            logger.info(f"Looking from {backfill_state.backfill_cursor + 1} onwards")
            project_preferences = (
                main_session.query(DbSeerProjectPreference)
                .filter(DbSeerProjectPreference.project_id > backfill_state.backfill_cursor)
                .order_by(DbSeerProjectPreference.project_id)
                .limit(MAX_QUERIED_PROJECT_IDS)
                .all()
            )

            if len(project_preferences) == 0:
                logger.info("No project preferences to backfill, looping")
                backfill_state.backfill_cursor = 0
                main_session.flush()

                logger.info(f"Looking from {backfill_state.backfill_cursor + 1} onwards")
                project_preferences = (
                    main_session.query(DbSeerProjectPreference)
                    .filter(DbSeerProjectPreference.project_id > backfill_state.backfill_cursor)
                    .order_by(DbSeerProjectPreference.project_id)
                    .limit(MAX_QUERIED_PROJECT_IDS)
                    .all()
                )

            if len(project_preferences) == 0:
                logger.info("No project preferences to backfill, done")
                return

            logger.info(
                f"Found {len(project_preferences)} project preferences to look at, starting from {project_preferences[0].project_id} to {project_preferences[-1].project_id}"
            )

            for project_preference in project_preferences:
                for repo in project_preference.repositories:
                    repo_definition = RepoDefinition.model_validate(repo)
                    repo_client = RepoClient.from_repo_definition(
                        repo_definition,
                        RepoClientType.READ,
                    )
                    blob_name = RepoManager.make_blob_name(
                        project_preference.organization_id,
                        repo_definition.provider,
                        repo_definition.owner,
                        repo_definition.name,
                        repo_definition.external_id,
                    )
                    count = (
                        main_session.query(DbSeerRepoArchive)
                        .filter(
                            DbSeerRepoArchive.organization_id == project_preference.organization_id,
                            DbSeerRepoArchive.bucket_name == RepoManager.get_bucket_name(),
                            DbSeerRepoArchive.blob_path == blob_name,
                        )
                        .count()
                    )

                    if count > 0:
                        logger.info(
                            f"Repository {repo_definition.full_name} for org {project_preference.organization_id} already has an archive, skipping."
                        )
                        continue

                    repo_key = (
                        project_preference.organization_id,
                        repo_definition.provider,
                        repo_definition.external_id,
                    )

                    if repo_key in processed_repos:
                        logger.info(
                            f"Repository {repo_definition.full_name} for org {project_preference.organization_id} already added to backfill jobs, skipping."
                        )
                        continue

                    backfill_jobs.append(
                        BackfillJob(
                            organization_id=project_preference.organization_id,
                            repo_definition=repo_definition,
                            scaled_time_limit=repo_client.get_scaled_time_limit(),
                        )
                    )
                    processed_repos.add(repo_key)

                if len(backfill_jobs) >= THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING:
                    break

            backfill_state.backfill_cursor = project_preference.project_id
            main_session.flush()

            for backfill_job in backfill_jobs:
                existing_backfill_job = (
                    main_session.query(DbSeerBackfillJob)
                    .filter(
                        DbSeerBackfillJob.organization_id == backfill_job.organization_id,
                        DbSeerBackfillJob.repo_provider == backfill_job.repo_definition.provider,
                        DbSeerBackfillJob.repo_external_id
                        == backfill_job.repo_definition.external_id,
                    )
                    .first()
                )
                if existing_backfill_job:
                    if not existing_backfill_job.failed_at and (
                        existing_backfill_job.started_at
                        and ensure_timezone_aware(existing_backfill_job.started_at)
                        < datetime.datetime.now(datetime.UTC) - timedelta(hours=1)
                    ):
                        logger.info(
                            f"Backfill job {existing_backfill_job.id} for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id} already started, skipping."
                        )
                        continue
                    else:

                        logger.info(
                            f"Deleting old backfill failed job {existing_backfill_job.id} for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id}"
                        )
                        main_session.delete(existing_backfill_job)
                        main_session.flush()

                db_backfill_job = DbSeerBackfillJob(
                    organization_id=backfill_job.organization_id,
                    repo_provider=backfill_job.repo_definition.provider,
                    repo_external_id=backfill_job.repo_definition.external_id,
                )

                main_session.add(db_backfill_job)
                main_session.flush()

                backfill_job.backfill_job_id = db_backfill_job.id

            main_session.commit()

            logger.info(f"Queueing {len(backfill_jobs)} backfill jobs")

    # Queue all jobs after transaction is complete
    for backfill_job in backfill_jobs:
        run_backfill.apply_async(
            args=(backfill_job.model_dump(),),
            soft_time_limit=backfill_job.scaled_time_limit,
            time_limit=backfill_job.scaled_time_limit + 30,  # 30 second buffer for hard timeout
        )


@inject
def run_backfill_task(app_config: AppConfig = injected):
    run_backfill.apply_async(queue=app_config.CELERY_WORKER_QUEUE)


@celery_app.task(
    soft_time_limit=timedelta(minutes=15).total_seconds(),
    time_limit=timedelta(minutes=15, seconds=30).total_seconds(),
)
def run_backfill(backfill_job_dict: dict):
    backfill_job = BackfillJob.model_validate(backfill_job_dict)

    if not backfill_job.backfill_job_id:
        raise BackfillJobError("backfill_job_id is required")

    logger.info(
        f"Running backfill job {backfill_job.backfill_job_id} for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id} with time limit {backfill_job.scaled_time_limit}"
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
    soft_time_limit=timedelta(minutes=15).total_seconds(),
    time_limit=timedelta(minutes=17).total_seconds(),
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


@celery_app.task(
    soft_time_limit=timedelta(minutes=1).total_seconds(),
    time_limit=timedelta(minutes=1, seconds=10).total_seconds(),
)
def run_repo_sync():
    # Go through repo archives and make the repo is up to date
    repos_to_update: list[RepoSyncJob] = []

    with Session() as main_session:
        backfill_state = (
            main_session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
        )

        if not backfill_state:
            backfill_state = DbSeerBackfillState(id=1, repo_sync_cursor=0)
            main_session.add(backfill_state)
            main_session.flush()

        repo_archives = (
            main_session.query(DbSeerRepoArchive)
            .where(DbSeerRepoArchive.id > backfill_state.repo_sync_cursor)
            .order_by(DbSeerRepoArchive.id)
            .limit(MAX_QUERIED_REPO_ARCHIVES_PER_SYNC)
            .all()
        )

        for repo_archive in repo_archives:
            repo_client = RepoClient.from_repo_definition(
                RepoManager.get_repo_definition_from_blob_name(repo_archive.blob_path),
            )
            commit_timestamp = repo_client.get_current_commit_info(repo_archive.commit_sha)[
                "timestamp"
            ]
            if commit_timestamp < datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7):
                logger.info(
                    f"Repo {repo_archive.blob_path} has a commit older than 7 days, needs to update"
                )
                repos_to_update.append(
                    RepoSyncJob(
                        archive_id=repo_archive.id,
                        scaled_time_limit=repo_client.get_scaled_time_limit(),
                    )
                )

        backfill_state.repo_sync_cursor = repo_archives[-1].id
        main_session.flush()

    for repo_sync_job in repos_to_update:
        run_repo_sync_for_repo_archive.apply_async(
            args=(repo_sync_job.model_dump(),),
            soft_time_limit=repo_sync_job.scaled_time_limit,
            time_limit=repo_sync_job.scaled_time_limit + 30,  # 30 second buffer for hard timeout
        )


@celery_app.task(
    soft_time_limit=timedelta(minutes=15).total_seconds(),
    time_limit=timedelta(minutes=15, seconds=30).total_seconds(),
)
def run_repo_sync_for_repo_archive(repo_sync_job_dict: dict):
    repo_sync_job = RepoSyncJob.model_validate(repo_sync_job_dict)

    with Session() as main_session:
        repo_archive = (
            main_session.query(DbSeerRepoArchive)
            .filter(DbSeerRepoArchive.id == repo_sync_job.archive_id)
            .first()
        )

        if not repo_archive:
            raise RepoSyncJobError("repo archive not found")

        repo_client = RepoClient.from_repo_definition(
            RepoManager.get_repo_definition_from_blob_name(repo_archive.blob_path),
        )
        repo_manager = RepoManager(
            repo_client, organization_id=repo_archive.organization_id, force_gcs=True
        )
        repo_manager.update_repo_archive()

        logger.info("Repo sync job done.")
