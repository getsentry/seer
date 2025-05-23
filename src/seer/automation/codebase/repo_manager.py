import datetime
import logging
import os
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable

import git
import sentry_sdk
from google.cloud import storage  # type:ignore
from sqlalchemy.orm import Session as SQLAlchemySession

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.utils import cleanup_dir, ensure_timezone_aware
from seer.configuration import AppConfig
from seer.db import DbSeerRepoArchive, Session
from seer.dependency_injection import copy_modules_initializer, inject, injected

logger = logging.getLogger(__name__)

# Minimum seconds between consecutive liveness probe updates
LIVENESS_UPDATE_INTERVAL = 5.0

UPLOAD_LOCK_TIMEOUT_MINUTES = 10


class RepoInitializationError(RuntimeError):
    """
    An error that occurs during the initialization of a repository.
    """

    pass


class RepoManager:
    """
    A client for downloading and syncing git repositories using GitPython.
    """

    repo_client: RepoClient
    git_repo: git.Repo | None
    tmp_dir: str
    repo_path: str
    initialization_future: Future | None
    is_cancelled: bool

    organization_id: int | None
    project_id: int | None

    _use_gcs: bool
    _last_liveness_update: float
    _trigger_liveness_probe: Callable[[], None] | None

    def __init__(
        self,
        repo_client: RepoClient,
        *,
        organization_id: int | None = None,
        project_id: int | None = None,
        trigger_liveness_probe: Callable[[], None] | None = None,
        force_gcs: bool = False,
    ):
        """
        Initialize the local git client.
        """
        self.repo_client = repo_client
        self.git_repo = None
        self.tmp_dir = tempfile.mkdtemp(
            prefix=f"{self.repo_client.repo_owner}-{self.repo_client.repo_name}"
        )
        self.repo_path = os.path.join(self.tmp_dir, "repo")
        self.initialization_future = None

        self.organization_id = organization_id
        self.project_id = project_id

        self._trigger_liveness_probe = trigger_liveness_probe
        self.is_cancelled = False
        self._last_liveness_update = 0.0

        self._use_gcs = (
            organization_id == 1 and project_id == 6178942
        ) or force_gcs  # TODO: Hardcoded ONLY for Seer and for backfill calls

    @staticmethod
    def make_blob_name(
        organization_id: int, provider: str, repo_owner: str, repo_name: str, repo_external_id: str
    ):
        return (
            f"repos/{organization_id}/{provider}/{repo_owner}/{repo_name}_{repo_external_id}.tar.gz"
        )

    @staticmethod
    @inject
    def get_bucket_name(config: AppConfig = injected):
        if not config.CODEBASE_GCS_STORAGE_BUCKET:
            raise RepoInitializationError("CODEBASE_GCS_STORAGE_BUCKET is not set, can't use GCS")

        return config.CODEBASE_GCS_STORAGE_BUCKET

    @property
    def is_available(self):
        return self.git_repo is not None and self.initialization_future is None

    @property
    def blob_name(self):
        return self.make_blob_name(
            self.organization_id,
            self.repo_client.provider,
            self.repo_client.repo_owner,
            self.repo_client.repo_name,
            self.repo_client.repo_external_id,
        )

    def initialize_in_background(self):
        logger.info(f"Creating initialize task for repo {self.repo_client.repo_full_name}")
        if self.initialization_future is not None:
            raise RepoInitializationError(
                f"Repo {self.repo_client.repo_full_name} is already being initialized"
            )

        self.initialization_future = ThreadPoolExecutor(
            1, initializer=copy_modules_initializer()
        ).submit(self.initialize)

    @sentry_sdk.trace
    def initialize(self):
        logger.info(f"Initializing repo {self.repo_client.repo_full_name}")

        try:
            if self.is_cancelled:
                return

            db_archive_entry: DbSeerRepoArchive | None = None
            if not self._use_gcs or not (db_archive_entry := self.gcs_archive_exists()):
                self._clone_repo()

            else:
                logger.info(
                    f"Using repository archive from GCS for {self.repo_client.repo_full_name}: {self.get_bucket_name()}/{self.blob_name}"
                )
                self.download_from_gcs()

            if self.is_cancelled:
                return

            self._sync_repo()

            if self.is_cancelled:
                return

            if db_archive_entry:
                original_timestamp = self.repo_client.get_current_commit_info(
                    db_archive_entry.commit_sha
                )["timestamp"]
                new_commit_time = self.repo_client.get_current_commit_info(
                    self.repo_client.base_commit_sha
                )["timestamp"]

                if new_commit_time > original_timestamp:
                    logger.info(
                        f"Repository {self.repo_client.repo_full_name} after syncing is in a newer state than before syncing",
                    )
                    if self._use_gcs:
                        logger.info(
                            f"Uploading newer state of repository {self.repo_client.repo_full_name} to GCS"
                        )

                        # Upload to GCS in a separate thread, don't wait for it.
                        ThreadPoolExecutor(
                            max_workers=1, initializer=copy_modules_initializer()
                        ).submit(self.upload_to_gcs)
                else:
                    logger.info(
                        f"Repository {self.repo_client.repo_full_name} after syncing is in an older state than before syncing",
                    )
            elif self._use_gcs:
                ThreadPoolExecutor(max_workers=1, initializer=copy_modules_initializer()).submit(
                    self.upload_to_gcs
                )
        except Exception:
            logger.exception(
                "Failed to initialize repo", extra={"repo": self.repo_client.repo_full_name}
            )
            self.cleanup()
            raise
        finally:
            self.initialization_future = None

            if self.is_cancelled:
                self.cleanup()

    def initialize_archive_for_backfill(self):
        try:
            self._clone_repo()
            self._sync_repo()
            self.upload_to_gcs(copy_repo=False)  # Don't copy for backfill to save time
        except Exception:
            logger.exception(
                "Failed to initialize repo archive during backfill",
                extra={"repo": self.repo_client.repo_full_name},
            )
            self.cleanup()
            raise

    @sentry_sdk.trace
    def _clone_repo(self) -> str:
        """
        Clone a repository to a local temporary directory.
        """
        repo_clone_url = self.repo_client.get_clone_url_with_auth()

        try:
            logger.info(
                f"Cloning repository {self.repo_client.repo_full_name} to {self.repo_path} with depth=1"
            )

            cleanup_dir(self.repo_path)

            start_time = time.time()
            self.git_repo = git.Repo.clone_from(
                repo_clone_url,
                self.repo_path,
                progress=lambda *args, **kwargs: self._throttled_liveness_probe(),
                depth=1,
                no_checkout=True,
            )
            end_time = time.time()
            logger.info(
                f"Cloned repository {self.repo_client.repo_full_name} in {end_time - start_time} seconds"
            )

            return self.repo_path
        except Exception:
            logger.exception(
                "Failed to clone repository", extra={"repo": self.repo_client.repo_full_name}
            )
            self.git_repo = None  # clear the repo to fail the available check
            raise

    @sentry_sdk.trace
    def _sync_repo(self):
        """
        Ensure the repository is up to date with only the target commit.
        """
        logger.info(f"Syncing repository {self.repo_client.repo_full_name}")

        try:
            start_time = time.time()
            commit_sha = self.repo_client.base_commit_sha

            # Reset any perceived local changes first
            self.git_repo.git.reset("--hard")
            # Clean any untracked files
            self.git_repo.git.clean("-fdx")

            auth_url = self.repo_client.get_clone_url_with_auth()

            # Fetch only the specific commit
            self.git_repo.git.execute(["git", "fetch", "--depth=1", auth_url, commit_sha])

            # Force checkout to avoid the "local changes" error
            logger.info(f"Checking out commit {commit_sha}")
            self.git_repo.git.checkout(commit_sha, force=True)

            end_time = time.time()
            logger.info(
                f"Checked out repo {self.repo_client.repo_full_name} to commit {commit_sha} in {end_time - start_time} seconds"
            )

            if self._use_gcs:
                self._verify_repo_state()
        except Exception:
            logger.exception(
                "Failed to sync repository", extra={"repo": self.repo_client.repo_full_name}
            )
            self.git_repo = None  # clear the repo to fail the available check

    def _verify_repo_state(self, repo_path: str | None = None):
        """
        Verify the repository state to ensure it is in the expected state at the right commit. Only use this for Debugging because it's slow.
        Should silently error to Sentry.
        """
        if repo_path is None:
            repo_path = self.repo_path

        if self.is_cancelled:
            return logger.exception(
                RepoInitializationError("Repository has been cancelled"),
                extra={"repo": self.repo_client.repo_full_name},
            )

        if self.git_repo is None:
            return logger.exception(
                RepoInitializationError("Repository is not cloned"),
                extra={"repo": self.repo_client.repo_full_name},
            )

        if self.git_repo.head.commit.hexsha != self.repo_client.base_commit_sha:
            return logger.exception(
                RepoInitializationError("Repository is not at the right commit"),
                extra={"repo": self.repo_client.repo_full_name},
            )

        invalid_file_paths = set()
        repo_file_paths = self.repo_client.get_valid_file_paths()
        for file_path in repo_file_paths:
            if not os.path.exists(os.path.join(repo_path, file_path)):
                invalid_file_paths.add(file_path)

        if invalid_file_paths:
            logger.exception(
                RepoInitializationError("Repository failed validation and is missing files"),
                extra={
                    "repo": self.repo_client.repo_full_name,
                    "invalid_file_paths": invalid_file_paths,
                },
            )
        else:
            logger.info(
                f"Repository {self.repo_client.repo_full_name} passed validation with no missing files",
            )

    def _throttled_liveness_probe(self):
        """
        Triggers the liveness probe only after a minimum time interval has elapsed since the last update.
        """
        current_time = time.time()
        if (
            current_time - self._last_liveness_update >= LIVENESS_UPDATE_INTERVAL
            and self._trigger_liveness_probe is not None
        ):
            self._trigger_liveness_probe()
            self._last_liveness_update = current_time

    def _copy_repo(self, target_folder: str = "copied_repo"):
        """
        Copy the repository to a new directory.
        """
        logger.info(f"Copying repository from {self.repo_path} to {self.tmp_dir}/{target_folder}")
        start_time = time.time()
        target_path = os.path.join(self.tmp_dir, target_folder)
        shutil.copytree(self.repo_path, target_path, symlinks=True)
        end_time = time.time()
        logger.info(f"Copied repository in {end_time - start_time} seconds")
        return target_path

    def _prune_repo(self, *, repo_path: str | None = None):
        """
        Prune the repository to only include the target commit.
        """
        if repo_path is None:
            repo_path = self.repo_path

        git_repo = git.Repo(repo_path)

        # Validate that the repo is initialized
        if not git_repo.git.execute(
            command=["git", "status"],
            stdout_as_string=True,
            with_extended_output=False,
            as_process=False,
        ):
            raise RepoInitializationError("Repository is not initialized")

        commit_sha = self.repo_client.base_commit_sha

        # Remove all remote branches except the one we need
        logger.info("Pruning unnecessary references")
        try:
            refs = git_repo.git.execute(
                ["git", "show-ref"],
                stdout_as_string=True,
                with_extended_output=False,
                as_process=False,
            )
            if refs:  # Only process if we have refs
                for ref in refs.split("\n"):
                    if ref and commit_sha not in ref:
                        ref_name = ref.split()[1]
                        try:
                            git_repo.git.execute(["git", "update-ref", "-d", ref_name])
                        except git.GitCommandError:
                            logger.debug(f"Could not delete reference {ref_name}")
        except git.GitCommandError:
            # Git show-ref returns exit code 1 if no refs exist, which is not necessarily an error
            logger.info("No references found in the repository")

        # Clean up completely - expire reflog, remove unreachable objects
        logger.info("Cleaning Git repository to minimal state")
        git_repo.git.execute(["git", "reflog", "expire", "--expire=now", "--all"])
        git_repo.git.execute(["git", "gc", "--prune=now"])

        # Remove all remotes
        git_repo.git.execute(["git", "remote", "remove", "origin"])

    @sentry_sdk.trace
    def upload_to_gcs(self, *, copy_repo: bool = True):
        """
        Upload the repository from the cloned tmp directory to GCS.
        This method uses a thread to perform the upload without blocking.

        Args:
            copy_repo: Whether to copy the repository before uploading. Defaults to True.
                       Set to False for backfill initialization to avoid unnecessary copying.
        """
        repo_path_to_use = None
        if copy_repo:
            repo_path_to_use = self._copy_repo()
            self._verify_repo_state(repo_path=repo_path_to_use)
            self._prune_repo(repo_path=repo_path_to_use)
        else:
            # Use the existing repo path directly without copying
            repo_path_to_use = self.repo_path
            self._verify_repo_state(repo_path=repo_path_to_use)
            self._prune_repo(repo_path=repo_path_to_use)

        with Session() as session:
            existing_repo_archive = self.get_db_archive_entry(session)

            if existing_repo_archive is None:
                repo_archive = DbSeerRepoArchive(
                    organization_id=self.organization_id,
                    bucket_name=self.get_bucket_name(),
                    blob_path=self.blob_name,
                    commit_sha=self.repo_client.base_commit_sha,
                )
                session.add(repo_archive)
            else:
                existing_repo_archive.commit_sha = self.repo_client.base_commit_sha

            session.commit()

        with self.upload_lock():
            # Create a temporary tar.gz file of the repository
            temp_tarfile = os.path.join(self.tmp_dir, "upload_repo_archive.tar.gz")
            try:
                logger.info(f"Creating tar archive of repository at {repo_path_to_use}")
                with tarfile.open(temp_tarfile, "w:gz") as tar:
                    for item in os.listdir(repo_path_to_use):
                        item_path = os.path.join(repo_path_to_use, item)
                        tar.add(item_path, arcname=item)

                if self.is_cancelled:
                    return self.cleanup()

                logger.info(
                    f"Uploading repository archive to GCS: {self.get_bucket_name()}/{self.blob_name}"
                )

                storage_client = storage.Client()
                bucket = storage_client.bucket(self.get_bucket_name())
                blob = bucket.blob(self.blob_name)
                blob.upload_from_filename(temp_tarfile)

                logger.info(
                    f"Successfully uploaded repository to GCS: {self.get_bucket_name()}/{self.blob_name}"
                )
            except Exception:
                logger.exception("Failed to upload repository to GCS")
                raise
            finally:
                # Clean up the temporary tar file
                if os.path.exists(temp_tarfile):
                    os.unlink(temp_tarfile)

                # Clean up the copied repo if we made one
                if (
                    copy_repo
                    and os.path.exists(repo_path_to_use)
                    and repo_path_to_use != self.repo_path
                ):
                    shutil.rmtree(repo_path_to_use)

    def get_db_archive_entry(self, session: SQLAlchemySession):
        if self.organization_id is None:
            raise RepoInitializationError("Organization ID is not set, can't get db archive entry")

        return (
            session.query(DbSeerRepoArchive)
            .filter(
                DbSeerRepoArchive.organization_id == self.organization_id,
                DbSeerRepoArchive.bucket_name == self.get_bucket_name(),
                DbSeerRepoArchive.blob_path == self.blob_name,
            )
            .first()
        )

    def gcs_archive_exists(self) -> DbSeerRepoArchive | None:
        """
        Check if an archive exists in GCS for this repository.

        Returns:
            bool: True if the archive exists, False otherwise
        """
        try:
            # Check if the archive exists in GCS
            logger.info(
                f"Checking if repository archive exists in GCS: {self.get_bucket_name()}/{self.blob_name}"
            )

            with Session() as session:
                if self.get_db_archive_entry(session) is None:
                    return None

            storage_client = storage.Client()
            bucket = storage_client.bucket(self.get_bucket_name())
            blob = bucket.blob(self.blob_name)

            exists = blob.exists()

            if exists:
                logger.info(
                    f"Repository archive found in GCS: {self.get_bucket_name()}/{self.blob_name}"
                )
            else:
                logger.info(
                    f"Repository archive not found in GCS: {self.get_bucket_name()}/{self.blob_name}"
                )

            if exists:
                with Session() as session:
                    return self.get_db_archive_entry(session)
            else:
                return None
        except Exception:
            logger.exception("Error checking if repository archive exists in GCS")
            return None

    @sentry_sdk.trace
    def download_from_gcs(self, chunk_size: int = 128 * 1024 * 1024, max_workers: int = 8):
        """
        Download the repository from GCS to the cloned tmp directory by slicing the blob into parallel chunks.

        Args:
            chunk_size: Size in bytes for each download segment (default: 128 * 1024 * 1024 or 128MB).
            max_workers: Maximum number of concurrent download threads to use (default: 8).
        """

        # Create a temporary file within self.tmp_dir to download the tar.gz
        temp_tarfile = os.path.join(self.tmp_dir, "repo_archive.tar.gz")

        try:
            # Download the tar file from GCS
            logger.info(
                f"Downloading repository archive from GCS: {self.get_bucket_name()}/{self.blob_name}"
            )
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.get_bucket_name())
            blob = bucket.blob(self.blob_name)

            if not blob.exists():
                raise FileNotFoundError(
                    f"Repository archive not found in GCS: {self.get_bucket_name()}/{self.blob_name}"
                )

            # Download the blob in parallel
            start_time = time.time()
            # Reload metadata to get blob size
            blob.reload()
            total_bytes = blob.size
            # Pre-allocate file
            with open(temp_tarfile, "wb") as f:
                f.truncate(total_bytes)
            # Prepare byte ranges for each chunk
            ranges = [
                (i, min(i + chunk_size - 1, total_bytes - 1))
                for i in range(0, total_bytes, chunk_size)
            ]

            def _download_range(start, end):
                data = blob.download_as_bytes(start=start, end=end)
                with open(temp_tarfile, "r+b") as f2:
                    f2.seek(start)
                    f2.write(data)

            # Fetch segments in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_download_range, start, end) for start, end in ranges]
                for future in futures:
                    future.result()
            end_time = time.time()
            logger.info(
                f"Downloaded repository archive for {self.repo_client.repo_full_name} in {end_time - start_time} seconds using {len(ranges)} segments"
            )

            # Clean up existing repo path before extracting
            cleanup_dir(self.repo_path)
            os.makedirs(self.repo_path, exist_ok=True)

            # Extract the tar file to the repo path using simple sequential extraction
            logger.info(f"Extracting repository archive to {self.repo_path}")
            with tarfile.open(temp_tarfile, "r:gz") as tar:
                # Get all members of the archive
                members = tar.getmembers()
                # Strip the top directory from paths and validate paths
                safe_members = []
                for member in members:
                    # Prevent path traversal attacks
                    if member.name.startswith("copied_repo/"):
                        member.name = member.name.replace("copied_repo/", "", 1)
                    else:
                        member.name = member.name.replace("copied_repo", "", 1)

                    # Ensure the path doesn't contain dangerous patterns like "../"
                    if ".." not in member.name and not os.path.isabs(member.name):
                        safe_members.append(member)
                    else:
                        logger.warning(f"Skipping potentially unsafe path: {member.name}")

                # Extract with modified and validated paths
                tar.extractall(path=self.repo_path, members=safe_members)

            logger.info(f"Extracted repository archive to {self.repo_path}")

            # Do a debug ls of the repo path, to ensure the archive was extracted correctly
            logger.debug(f"Debug ls of repo path {self.repo_path}: {os.listdir(self.repo_path)}")

            self.git_repo = git.Repo(self.repo_path)

            logger.info(f"Successfully downloaded repository from GCS to {self.repo_path}")
        except Exception:
            logger.exception("Failed to download repository from GCS")
            raise
        finally:
            # Clean up the temporary files
            if os.path.exists(temp_tarfile):
                os.unlink(temp_tarfile)

    @contextmanager
    def upload_lock(self):
        did_lock = False
        try:
            with Session() as session:
                repo_archive = self.get_db_archive_entry(session)
                if repo_archive and (
                    # Not locked
                    not repo_archive.upload_locked_at
                    # Expired
                    or ensure_timezone_aware(repo_archive.upload_locked_at)
                    < datetime.datetime.now(datetime.UTC)
                    - datetime.timedelta(minutes=UPLOAD_LOCK_TIMEOUT_MINUTES)
                ):
                    repo_archive.upload_locked_at = datetime.datetime.now(datetime.UTC)
                    session.commit()
                    did_lock = True
                else:
                    logger.info(
                        f"Repository is already locked for upload, was locked at {repo_archive.upload_locked_at} ({datetime.datetime.now(datetime.UTC) - repo_archive.upload_locked_at})",
                    )
                    raise RepoInitializationError("Repository is already locked for upload")

            yield
        finally:
            # only clear the lock if we actually set it
            if did_lock:
                with Session() as session:
                    repo_archive = self.get_db_archive_entry(session)
                    if repo_archive:
                        repo_archive.upload_locked_at = None
                        session.commit()

    def mark_as_timed_out(self):
        logger.error(
            "Repo manager timed out",
            extra={"repo": self.repo_client.repo_full_name},
        )

        if self.initialization_future:
            # Have the thread deal with cleanup
            self.is_cancelled = True
        else:
            # Do it now
            self.cleanup()

    def cleanup(self):
        """
        Clean up the temporary directory if it exists, and mark the repo as cancelled.
        """
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            try:
                # Cleaning up with the tmp_dir itself ensures that we don't leave the archives or the copied repo folder behind
                cleanup_dir(self.tmp_dir)
            except Exception as e:
                logger.error(f"Error during repo cleanup, but continuing: {e}")
            finally:
                # Ensure we null out paths even if cleanup fails
                self.repo_path = None
                self.tmp_dir = None

        # always marked as cancelled and clear out the repo if we're here
        self.is_cancelled = True
        self.git_repo = None

        logger.info(f"Cleaned up repo for {self.repo_client.repo_full_name}")
