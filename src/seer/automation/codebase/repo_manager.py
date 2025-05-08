import asyncio
import logging
import os
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import git
from google.cloud import storage

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.utils import cleanup_dir

logger = logging.getLogger(__name__)


class RepoManager:
    """
    A client for interacting with local git repositories using GitPython.
    """

    def __init__(
        self, repo_client: RepoClient, trigger_liveness_probe: Callable[[], None] | None = None
    ):
        """
        Initialize the local git client.
        """
        self.repo_client = repo_client
        self.git_repo = None
        self.tmp_dir = tempfile.mkdtemp(
            prefix=f"{self.repo_client.repo_owner}-{self.repo_client.repo_name}_{self.repo_client.base_commit_sha}"
        )
        self.repo_path = os.path.join(self.tmp_dir, "repo")
        self.initialization_future = None
        self._trigger_liveness_probe = trigger_liveness_probe

    @property
    def is_available(self):
        return self.git_repo is not None and self.initialization_future is None

    @property
    def blob_name(self):
        """
        Get the blob name for the repository.
        """
        return f"repos/{self.repo_client.repo_owner}/{self.repo_client.repo_name}.tar.gz"

    @property
    def bucket_name(self):
        """
        Get the bucket name for the repository.
        """
        return os.environ.get("AUTOFIX_REPOS_GCS_BUCKET", "autofix-repositories-local")

    def initialize_in_background(self, gcs_enabled: bool = False):
        logger.info(f"Creating initialize task for repo {self.repo_client.repo_full_name}")
        self.initialization_future = ThreadPoolExecutor(1).submit(
            self.initialize, gcs_enabled=gcs_enabled
        )

    def initialize(self, gcs_enabled: bool = False):
        logger.info(f"Initializing repo {self.repo_client.repo_full_name}")

        try:
            if gcs_enabled and self.gcs_archive_exists():
                self.download_from_gcs()
            else:
                self._clone_repo()

                if gcs_enabled:
                    # Only upload to GCS if this is a fresh new clone of the repo.
                    self.upload_to_gcs()

            logger.info(f"Repo {self.repo_client.repo_full_name} ready at {self.repo_path}")

            self._sync_repo()
        except Exception as e:
            logger.error(f"Failed to initialize repo {self.repo_client.repo_full_name}: {e}")
            raise
        finally:
            self.initialization_future = None

    def _clone_repo(self) -> str:
        """
        Clone a repository to a local temporary directory.

        Args:
            repo_clone_url: The URL of the repository to clone

        Returns:
            The path to the cloned repository
        """

        repo_clone_url = self.repo_client.get_clone_url_with_auth()

        try:
            logger.info(f"Cloning repository {repo_clone_url} to {self.repo_path} with depth=1")

            cleanup_dir(self.repo_path)

            start_time = time.time()
            self.git_repo = git.Repo.clone_from(
                repo_clone_url,
                self.repo_path,
                progress=lambda *args, **kwargs: self._trigger_liveness_probe(),
                depth=1,
            )
            end_time = time.time()
            logger.info(f"Cloned repository in {end_time - start_time} seconds")

            return self.repo_path
        except git.GitCommandError as e:
            self.cleanup()
            logger.error(f"Failed to clone repository: {e}")
            raise

    def _sync_repo(self):
        """
        Ensure the repository is up to date with only the target commit.
        """
        logger.info(f"Syncing repository {self.repo_client.repo_full_name}")

        try:
            start_time = time.time()
            commit_sha = self.repo_client.base_commit_sha

            # Fetch only the specific commit
            self.git_repo.git.execute(["git", "fetch", "--depth=1", "origin", commit_sha])
            self.git_repo.git.checkout(commit_sha)

            end_time = time.time()
            logger.info(
                f"Checked out repo {self.repo_client.repo_full_name} to commit {commit_sha} in {end_time - start_time} seconds"
            )
        except Exception as e:
            logger.error(f"Failed to sync repository {self.repo_client.repo_full_name}: {e}")
            self.git_repo = None  # clear the repo to fail the available check
            raise

    def _prune_repo(self):
        """
        Prune the repository to only include the target commit.
        """
        commit_sha = self.repo_client.base_commit_sha

        # Remove all remote branches except the one we need
        logger.info("Pruning unnecessary references")
        for ref in self.git_repo.git.execute(["git", "show-ref"]).split("\n"):
            if ref and commit_sha not in ref:
                ref_name = ref.split()[1]
                try:
                    self.git_repo.git.execute(["git", "update-ref", "-d", ref_name])
                except git.GitCommandError:
                    logger.debug(f"Could not delete reference {ref_name}")

        # Clean up completely - expire reflog, remove unreachable objects
        logger.info("Cleaning Git repository to minimal state")
        self.git_repo.git.execute(["git", "reflog", "expire", "--expire=now", "--all"])
        self.git_repo.git.execute(["git", "gc", "--prune=now", "--aggressive"])

    def upload_to_gcs(self):
        """
        Upload the repository from the cloned tmp directory to GCS.
        This method is a synchronous wrapper around the async _upload_to_gcs_async method.
        """
        self._prune_repo()

        # Run the async upload in the asyncio event loop
        asyncio.run(self._upload_to_gcs_async())

    async def _upload_to_gcs_async(self):
        """
        Asynchronous implementation of the repository upload to GCS.
        """
        # Create a temporary tar.gz file of the repository
        temp_tarfile = os.path.join(self.tmp_dir, "repo_archive.tar.gz")
        try:
            logger.info(f"Creating tar archive of repository at {self.repo_path}")
            # Create tarfile synchronously since it's IO-bound and doesn't benefit much from async
            with tarfile.open(temp_tarfile, "w:gz") as tar:
                tar.add(self.repo_path, arcname=os.path.basename(self.repo_path))

            # Upload the tar file to GCS asynchronously
            logger.info(f"Uploading repository archive to GCS: {self.bucket_name}/{self.blob_name}")

            # Run the GCS upload in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._do_gcs_upload, temp_tarfile)

            logger.info(
                f"Successfully uploaded repository to GCS: {self.bucket_name}/{self.blob_name}"
            )
        except Exception as e:
            logger.error(f"Failed to upload repository to GCS: {e}")
            raise
        finally:
            # Clean up the temporary tar file
            if os.path.exists(temp_tarfile):
                os.unlink(temp_tarfile)

    def _do_gcs_upload(self, temp_tarfile):
        """
        Perform the actual GCS upload operation.
        This method is designed to be run in a separate thread.

        Args:
            temp_tarfile: Path to the temporary tarfile to upload
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(self.blob_name)
        blob.upload_from_filename(temp_tarfile)

    def gcs_archive_exists(self):
        """
        Check if an archive exists in GCS for this repository.

        Returns:
            bool: True if the archive exists, False otherwise
        """
        try:
            # Check if the archive exists in GCS
            logger.info(
                f"Checking if repository archive exists in GCS: {self.bucket_name}/{self.blob_name}"
            )
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(self.blob_name)

            exists = blob.exists()

            if exists:
                logger.info(f"Repository archive found in GCS: {self.bucket_name}/{self.blob_name}")
            else:
                logger.info(
                    f"Repository archive not found in GCS: {self.bucket_name}/{self.blob_name}"
                )

            return exists
        except Exception as e:
            logger.error(f"Error checking if repository archive exists in GCS: {e}")
            return False

    def download_from_gcs(self):
        """
        Download the repository from GCS to the cloned tmp directory.

        Args:
            max_workers: Maximum number of concurrent extraction threads (default: 4)
                         [No longer used as extraction is now sequential]
        """

        # Create a temporary file within self.tmp_dir to download the tar.gz
        temp_tarfile = os.path.join(self.tmp_dir, "repo_archive.tar.gz")

        try:
            # Download the tar file from GCS
            logger.info(
                f"Downloading repository archive from GCS: {self.bucket_name}/{self.blob_name}"
            )
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(self.blob_name)

            if not blob.exists():
                logger.error(
                    f"Repository archive not found in GCS: {self.bucket_name}/{self.blob_name}"
                )
                raise FileNotFoundError(
                    f"Repository archive not found in GCS: {self.bucket_name}/{self.blob_name}"
                )

            # Download the blob to the temporary file with larger chunk size for faster download
            start_time = time.time()
            blob.download_to_filename(temp_tarfile)
            end_time = time.time()
            logger.info(f"Downloaded repository archive in {end_time - start_time} seconds")

            # Clean up existing repo path before extracting
            cleanup_dir(self.repo_path)
            os.makedirs(self.repo_path, exist_ok=True)

            # Extract the tar file to the repo path using simple sequential extraction
            logger.info(f"Extracting repository archive to {self.repo_path}")
            with tarfile.open(temp_tarfile, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(self.repo_path))

            self.git_repo = git.Repo(self.repo_path)

            logger.info(f"Successfully downloaded repository from GCS to {self.repo_path}")
        except Exception as e:
            logger.error(f"Failed to download repository from GCS: {e}")
            raise
        finally:
            # Clean up the temporary files
            if os.path.exists(temp_tarfile):
                os.unlink(temp_tarfile)

    def cleanup(self):
        """
        Clean up the temporary directory if it exists.
        """
        if self.repo_path and os.path.exists(self.repo_path):
            cleanup_dir(self.repo_path)
            self.repo_path = None
            self.git_repo = None

        logger.info(f"Cleaned up local repo client for {self.repo_client.repo_full_name}")

    def __del__(self):
        """
        Ensure cleanup happens when the object is garbage collected.
        """
        self.cleanup()
