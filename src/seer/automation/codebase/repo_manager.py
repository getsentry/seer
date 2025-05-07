import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import git

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.utils import cleanup_dir

logger = logging.getLogger(__name__)


class RepoManager:
    """
    A client for downloading and syncing git repositories using GitPython.
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

    def initialize_in_background(self):
        logger.info(f"Creating initialize task for repo {self.repo_client.repo_full_name}")
        self.initialization_future = ThreadPoolExecutor(1).submit(self.initialize)

    def initialize(self):
        logger.info(f"Initializing repo {self.repo_client.repo_full_name}")

        try:
            self._clone_repo()

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
