import datetime
import logging
from contextlib import contextmanager
from datetime import timedelta

from pydantic import BaseModel
from sqlalchemy import and_, or_, text

from celery_app.app import celery_app
from seer.automation.codebase.repo_client import RepoClient, RepoClientType
from seer.automation.codebase.repo_manager import RepoManager
from seer.automation.codebase.utils import ensure_timezone_aware
from seer.automation.models import RepoDefinition
from seer.db import (
    DbSeerBackfillJob,
    DbSeerBackfillState,
    DbSeerProjectPreference,
    DbSeerRepoArchive,
    Session,
)

logger = logging.getLogger(__name__)

# These lock keys ensure that only one backfill or sync task runs at a time, given our current setup with multiple celerybeat instances.
BACKFILL_LOCK_KEY = 1234567890123456780
SYNC_LOCK_KEY = 1234567890123456781
CLEANUP_LOCK_KEY = 1234567890123456782

MAX_QUERIED_PROJECT_IDS = 250
THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING = 32

MAX_REPO_ARCHIVES_PER_SYNC = 32
REPO_ARCHIVE_UPDATE_INTERVAL = datetime.timedelta(days=7)
REPO_ARCHIVE_CLEANUP_INTERVAL = datetime.timedelta(days=30)
MAX_REPO_ARCHIVES_PER_CLEANUP = 50

# Temporarily limit processing to specific organization IDs
FLAGGED_ORG_IDS = {1}


@contextmanager
def acquire_lock(session, lock_key: int, lock_name: str):
    """
    Acquire a Postgres advisory lock for the backfill or sync task.
    """

    try:
        # pg_try_advisory_xact_lock returns true if lock acquired, false if not
        # This is a transaction-level lock that auto-releases when transaction ends
        got_lock = session.execute(
            text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": lock_key}
        ).scalar()

        if not got_lock:
            logger.info(f"Could not acquire {lock_name} lock, another process has it")
            yield False
        else:
            logger.info(f"Acquired {lock_name} lock")
            yield True

    except Exception:
        logger.exception(f"Error while managing {lock_name} lock")
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
    repo_full_name: str
    scaled_time_limit: float = timedelta(minutes=15).total_seconds()


class RepoSyncJobError(RuntimeError):
    pass


@celery_app.task(
    soft_time_limit=timedelta(minutes=1).total_seconds(),
    time_limit=timedelta(minutes=1, seconds=10).total_seconds(),
)
def collect_all_repos_for_backfill():
    """
    Collects repositories that need to be backfilled and queues backfill jobs for them.

    This task is the main orchestrator for the backfill process. It processes up to
    MAX_QUERIED_PROJECT_IDS project preferences per run and stops after
    THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING jobs are queued to prevent overwhelming the system.

    Returns:
        None - This task queues other tasks but doesn't return a value
    """
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

        # 1. Use PostgreSQL advisory lock to ensure only one instance runs at a time
        with acquire_lock(
            main_session, lock_key=BACKFILL_LOCK_KEY, lock_name="backfill"
        ) as got_lock:
            if not got_lock:
                return

            # If we get here, we have the lock
            # 2. Query project preferences in batches, starting from a cursor position
            logger.info(f"Looking from {backfill_state.backfill_cursor + 1} onwards")
            project_preferences = (
                main_session.query(DbSeerProjectPreference)
                .filter(
                    DbSeerProjectPreference.project_id > backfill_state.backfill_cursor,
                    DbSeerProjectPreference.organization_id.in_(
                        FLAGGED_ORG_IDS
                    ),  # Temporarily only run on flagged org ids
                )
                .order_by(DbSeerProjectPreference.project_id)
                .limit(MAX_QUERIED_PROJECT_IDS)
                .all()
            )

            if len(project_preferences) == 0:
                logger.info("No project preferences to backfill, looping")
                # Reset backfill cursor to start from beginning
                backfill_state.backfill_cursor = 0
                main_session.flush()

                logger.info(f"Looking from {backfill_state.backfill_cursor + 1} onwards")
                project_preferences = (
                    main_session.query(DbSeerProjectPreference)
                    .filter(
                        DbSeerProjectPreference.project_id > backfill_state.backfill_cursor,
                        DbSeerProjectPreference.organization_id.in_(
                            FLAGGED_ORG_IDS
                        ),  # Temporarily only run on flagged org ids
                    )
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

                    # 3. For each repository in the preferences, check if it already has an archive
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

                    # 4. Create BackfillJob entries for repositories that need backfilling
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

            # 5. Update the backfill cursor to track progress across runs
            backfill_state.backfill_cursor = project_preference.project_id
            main_session.flush()

            # 6. Clean up old/failed backfill jobs before creating new ones
            for backfill_job in backfill_jobs:
                # Use atomic delete with conditions to avoid race conditions
                deleted_count = (
                    main_session.query(DbSeerBackfillJob)
                    .filter(
                        DbSeerBackfillJob.organization_id == backfill_job.organization_id,
                        DbSeerBackfillJob.repo_provider == backfill_job.repo_definition.provider,
                        DbSeerBackfillJob.repo_external_id
                        == backfill_job.repo_definition.external_id,
                        # Only delete if failed OR if started more than 1 hour ago without completion
                        or_(
                            DbSeerBackfillJob.failed_at.isnot(None),
                            and_(
                                DbSeerBackfillJob.started_at.isnot(None),
                                DbSeerBackfillJob.completed_at.is_(None),
                                DbSeerBackfillJob.started_at
                                < datetime.datetime.now(datetime.UTC) - timedelta(hours=1),
                            ),
                        ),
                    )
                    .delete(synchronize_session=False)
                )

                # Check if there's still an active job (not deleted by the above operation)
                existing_active_job = (
                    main_session.query(DbSeerBackfillJob)
                    .filter(
                        DbSeerBackfillJob.organization_id == backfill_job.organization_id,
                        DbSeerBackfillJob.repo_provider == backfill_job.repo_definition.provider,
                        DbSeerBackfillJob.repo_external_id
                        == backfill_job.repo_definition.external_id,
                    )
                    .first()
                )

                if existing_active_job:
                    logger.info(
                        f"Backfill job for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id} is still active, skipping."
                    )
                    continue

                if deleted_count > 0:
                    logger.info(
                        f"Deleted {deleted_count} old backfill job(s) for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id}"
                    )

                # Create new job
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

    # 7. Queue individual backfill tasks for each repository
    for backfill_job in backfill_jobs:
        run_backfill.apply_async(
            args=(backfill_job.model_dump(),),
            soft_time_limit=backfill_job.scaled_time_limit,
            time_limit=backfill_job.scaled_time_limit + 30,  # 30 second buffer for hard timeout
        )


@celery_app.task(
    soft_time_limit=timedelta(minutes=15).total_seconds(),
    time_limit=timedelta(minutes=15, seconds=30).total_seconds(),
)
def run_backfill(backfill_job_dict: dict):
    """
    Executes a single repository backfill job to create an archive in Google Cloud Storage.

    This task performs the actual backfill work for a specific repository. The backfill process
    creates a compressed archive containing the repository at a specific commit, which can later
    be downloaded and used without needing to clone from the original source.

    Args:
        backfill_job_dict (dict): Serialized BackfillJob containing:
            - organization_id: The organization that owns the repository
            - repo_definition: Repository details (provider, owner, name, etc.)
            - backfill_job_id: Database ID of the backfill job record
            - scaled_time_limit: Time limit adjusted for repository complexity

    Raises:
        BackfillJobError: If the job is invalid, already processed, or not found
        Exception: Any other errors during the backfill process
    """
    # 1. Validate the backfill job parameters and ensure the job hasn't already been processed
    backfill_job = BackfillJob.model_validate(backfill_job_dict)

    if not backfill_job.backfill_job_id:
        raise BackfillJobError("backfill_job_id is required")

    logger.info(
        f"Running backfill job {backfill_job.backfill_job_id} for {backfill_job.repo_definition.full_name} of org {backfill_job.organization_id} with time limit {backfill_job.scaled_time_limit}"
    )

    # 2. Mark the job as started in the database to prevent duplicate processing
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
        # 3. Create a RepoClient and RepoManager for the specified repository
        repo_client = RepoClient.from_repo_definition(
            backfill_job.repo_definition, RepoClientType.READ
        )
        repo_manager = RepoManager(repo_client, organization_id=backfill_job.organization_id)

        # 4. Call initialize_archive_for_backfill() which clones, syncs, and uploads the archive
        repo_manager.initialize_archive_for_backfill()

        logger.info("Backfill job done.")

        # 5. Update the job status to completed upon success
        with Session() as session:
            session.query(DbSeerBackfillJob).filter(
                DbSeerBackfillJob.id == backfill_job.backfill_job_id
            ).update(
                {
                    DbSeerBackfillJob.completed_at: datetime.datetime.now(datetime.UTC),
                }
            )
            session.commit()

        # 6. Queue a verification task to test the created archive
        run_test_download_and_verify_backfill.apply_async(args=(backfill_job_dict,))
    except Exception:
        # 7. Handle failures by marking the job as failed and cleaning up resources
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
        # Only cleanup if repo_manager was successfully created
        if "repo_manager" in locals():
            repo_manager.cleanup()


@celery_app.task(
    soft_time_limit=timedelta(minutes=15).total_seconds(),
    time_limit=timedelta(minutes=17).total_seconds(),
)
def run_test_download_and_verify_backfill(backfill_job_dict: dict):
    """
    Verifies that a completed backfill job created a valid and accessible repository archive.

    This task is automatically queued after a successful backfill to ensure the archive
    was properly created and can be downloaded. This verification step is crucial to catch
    any issues with the archive creation process, such as corruption during upload,
    incomplete file transfers, or extraction problems.

    Args:
        backfill_job_dict (dict): Serialized BackfillJob containing:
            - organization_id: The organization that owns the repository
            - repo_definition: Repository details (provider, owner, name, etc.)
            - backfill_job_id: Database ID of the backfill job record

    Raises:
        BackfillJobError: If the job is not found or already verified
        Exception: Any errors during the verification process
    """
    # 1. Validate that the backfill job exists and hasn't already been verified
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

    # 2. Create a RepoClient and RepoManager for the repository
    repo_client = RepoClient.from_repo_definition(backfill_job.repo_definition, RepoClientType.READ)
    repo_manager = RepoManager(repo_client, organization_id=backfill_job.organization_id)

    # 3. Call initialize() which downloads, extracts, and verifies the archive
    repo_manager.initialize()

    logger.info("Backfill verification job done.")

    # 4. Mark the job as verified in the database upon successful verification
    with Session() as session:
        session.query(DbSeerBackfillJob).filter(
            DbSeerBackfillJob.id == backfill_job.backfill_job_id
        ).update(
            {
                DbSeerBackfillJob.verified_at: datetime.datetime.now(datetime.UTC),
            }
        )
        session.commit()

    # 5. Clean up any temporary files created during verification
    repo_manager.cleanup()


@celery_app.task(
    soft_time_limit=timedelta(minutes=1).total_seconds(),
    time_limit=timedelta(minutes=1, seconds=10).total_seconds(),
)
def run_repo_sync():
    """
    Identifies repository archives that need updating and queues sync jobs for them.

    This task maintains the freshness of repository archives by periodically updating them
    with the latest changes from the source repositories. This task runs periodically
    (typically daily) to ensure repository archives stay reasonably up-to-date without
    overwhelming the system.

    Returns:
        None - This task queues other tasks but doesn't return a value
    """
    # Go through repo archives and make the repo is up to date
    repos_to_update: list[RepoSyncJob] = []

    with Session() as main_session:
        # 1. Use a PostgreSQL advisory lock to ensure only one sync process runs at a time
        with acquire_lock(main_session, lock_key=SYNC_LOCK_KEY, lock_name="sync") as got_lock:
            if not got_lock:
                return

            # 2. Query for repository archives that haven't been updated within the interval (7 days)
            # 3. Order archives by their last update time to prioritize the oldest ones
            # 4. Limit processing to MAX_REPO_ARCHIVES_PER_SYNC (32) archives per run
            # Only sync repos that have been used (downloaded) within the last 7 days
            repo_archives = (
                main_session.query(DbSeerRepoArchive)
                .where(
                    and_(
                        DbSeerRepoArchive.updated_at.isnot(None),
                        DbSeerRepoArchive.updated_at
                        < datetime.datetime.now(datetime.UTC) - REPO_ARCHIVE_UPDATE_INTERVAL,
                        DbSeerRepoArchive.last_downloaded_at.isnot(None),
                        DbSeerRepoArchive.last_downloaded_at
                        >= datetime.datetime.now(datetime.UTC) - REPO_ARCHIVE_UPDATE_INTERVAL,
                        DbSeerRepoArchive.organization_id.in_(
                            FLAGGED_ORG_IDS
                        ),  # Temporarily only run on flagged org ids
                    )
                )
                .order_by(DbSeerRepoArchive.updated_at)
                .limit(MAX_REPO_ARCHIVES_PER_SYNC)
                .all()
            )

            # 5. For each archive, create a RepoSyncJob with appropriate time limits
            for repo_archive in repo_archives:
                try:
                    repo_client = RepoClient.from_repo_definition(
                        RepoDefinition.model_validate(repo_archive.repo_definition),
                        RepoClientType.READ,
                    )

                    repos_to_update.append(
                        RepoSyncJob(
                            archive_id=repo_archive.id,
                            repo_full_name=repo_client.repo_full_name,
                            scaled_time_limit=repo_client.get_scaled_time_limit(),
                        )
                    )
                except Exception as e:
                    # 6. Handle repositories that no longer exist by deleting their archives
                    if "Error getting repo via full name" in str(e):
                        logger.info(
                            f"Repo {repo_archive.blob_path} not found from github api, deleting repo archive"
                        )
                        main_session.delete(repo_archive)
                        main_session.flush()
                        continue

                    logger.exception(
                        "Failed to get repo_client for repo",
                        extra={"repo_archive_id": repo_archive.id},
                    )
                    continue

        main_session.commit()

    logger.info(f"Queueing {len(repos_to_update)} repo sync jobs")

    # 7. Queue individual sync tasks for each repository that needs updating
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
    """
    Updates a specific repository archive with the latest changes from the source repository.

    This task performs the actual synchronization work for a single repository archive.
    The sync process ensures that archived repositories stay current with their source
    repositories by periodically updating them with the latest commits.

    Args:
        repo_sync_job_dict (dict): Serialized RepoSyncJob containing:
            - archive_id: Database ID of the repository archive to update
            - repo_full_name: Full name of the repository (owner/name)
            - scaled_time_limit: Time limit adjusted for repository complexity

    Raises:
        RepoSyncJobError: If the repository archive is not found
        Exception: Any errors during the sync process
    """
    # 1. Validate the sync job parameters and retrieve the repository archive from the database
    repo_sync_job = RepoSyncJob.model_validate(repo_sync_job_dict)

    logger.info(
        f"Running repo sync for repo {repo_sync_job.repo_full_name} with time limit {repo_sync_job.scaled_time_limit}"
    )

    with Session() as main_session:
        repo_archive = (
            main_session.query(DbSeerRepoArchive)
            .filter(DbSeerRepoArchive.id == repo_sync_job.archive_id)
            .first()
        )

    if not repo_archive:
        raise RepoSyncJobError("repo archive not found")

    # 2. Check if the archive was recently updated to avoid unnecessary work
    updated_at = ensure_timezone_aware(repo_archive.updated_at)
    if (
        updated_at is not None
        and updated_at > datetime.datetime.now(datetime.UTC) - REPO_ARCHIVE_UPDATE_INTERVAL
    ):
        logger.info(
            f"Repo {repo_sync_job.repo_full_name} was last updated less than {REPO_ARCHIVE_UPDATE_INTERVAL} ago, skipping"
        )
        return

    # 3. Create a RepoClient and RepoManager for the repository
    repo_client = RepoClient.from_repo_definition(
        RepoDefinition.model_validate(repo_archive.repo_definition),
        RepoClientType.READ,
    )
    repo_manager = RepoManager(repo_client, organization_id=repo_archive.organization_id)
    try:
        # 4. Call update_repo_archive() which downloads, syncs, and re-uploads the archive
        repo_manager.update_repo_archive()
    except Exception as e:
        logger.info(f"DEBUG: Exception type: {type(e)}, Exception str: {str(e)}")
        # 5. Handle cases where the archive no longer exists in GCS by deleting the database record
        if "Repository archive not found in GCS" in str(e):
            logger.info(
                f"Repo {repo_sync_job.repo_full_name} not found in GCS, deleting repo archive"
            )
            with Session() as session:
                session.delete(repo_archive)
                session.commit()
            return

        # 6. Log errors and re-raise exceptions for proper error handling
        logger.exception(
            "Failed to update repo archive",
            extra={"repo_archive_id": repo_archive.id},
        )
        raise

    logger.info("Repo sync job done.")


@celery_app.task(
    soft_time_limit=timedelta(minutes=5).total_seconds(),
    time_limit=timedelta(minutes=5, seconds=30).total_seconds(),
)
def run_repo_archive_cleanup():
    """
    Removes repository archives that haven't been updated in the last 30 days.

    This task helps manage storage costs by cleaning up old repository archives that are
    no longer being actively maintained. It removes both the database records and the
    corresponding GCS blobs to free up storage space.

    The cleanup process:
    1. Uses an advisory lock to ensure only one cleanup process runs at a time
    2. Queries for archives older than REPO_ARCHIVE_CLEANUP_INTERVAL (30 days)
    3. Limits processing to MAX_REPO_ARCHIVES_PER_CLEANUP (50) archives per run
    4. Uses RepoManager.delete_archive() to remove both GCS blob and database record
    5. Logs the cleanup actions for monitoring and debugging

    Returns:
        None - This task performs cleanup but doesn't return a value
    """
    logger.info("Starting repository archive cleanup")

    archives_to_cleanup = []

    with Session() as main_session:
        # 1. Use a PostgreSQL advisory lock to ensure only one cleanup process runs at a time
        with acquire_lock(main_session, lock_key=CLEANUP_LOCK_KEY, lock_name="cleanup") as got_lock:
            if not got_lock:
                return

            # 2. Query for repository archives that haven't been updated within the cleanup interval (30 days)
            cutoff_date = datetime.datetime.now(datetime.UTC) - REPO_ARCHIVE_CLEANUP_INTERVAL

            repo_archives = (
                main_session.query(DbSeerRepoArchive)
                .where(
                    and_(
                        or_(
                            and_(
                                DbSeerRepoArchive.last_downloaded_at.isnot(None),
                                DbSeerRepoArchive.last_downloaded_at < cutoff_date,
                            ),
                            and_(
                                DbSeerRepoArchive.last_downloaded_at.is_(None),
                                DbSeerRepoArchive.updated_at < cutoff_date,
                            ),
                        ),
                        DbSeerRepoArchive.organization_id.in_(
                            FLAGGED_ORG_IDS
                        ),  # Temporarily only run on flagged org ids
                    )
                )
                .order_by(DbSeerRepoArchive.updated_at)  # Clean up oldest first
                .limit(MAX_REPO_ARCHIVES_PER_CLEANUP)
                .all()
            )

            logger.info(f"Found {len(repo_archives)} repository archives to clean up")

            # 3. For each archive, use RepoManager to delete both GCS blob and database record
            for repo_archive in repo_archives:
                try:
                    # Create a RepoClient and RepoManager for the repository
                    # TODO: What if they revoke access to this repo? We should still be able to delete the archive.
                    repo_client = RepoClient.from_repo_definition(
                        RepoDefinition.model_validate(repo_archive.repo_definition),
                        RepoClientType.READ,
                    )
                    repo_manager = RepoManager(
                        repo_client, organization_id=repo_archive.organization_id
                    )

                    # Use RepoManager's delete_archive method to handle both GCS and DB cleanup
                    repo_manager.delete_archive()
                    archives_to_cleanup.append(repo_archive.blob_path)

                except Exception as e:
                    logger.exception(
                        "Failed to cleanup repository archive",
                        extra={
                            "repo_archive_id": repo_archive.id,
                            "blob_path": repo_archive.blob_path,
                            "error": str(e),
                        },
                    )
                    # Continue with other archives even if one fails
                    continue

        # Note: No need to commit here since RepoManager.delete_archive() handles its own session

    logger.info(f"Successfully cleaned up {len(archives_to_cleanup)} repository archives")
    if archives_to_cleanup:
        logger.info(f"Cleaned up archives: {archives_to_cleanup}")
