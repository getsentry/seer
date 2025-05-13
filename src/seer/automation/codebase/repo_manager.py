import logging
import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import git
import sentry_sdk

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.utils import cleanup_dir

logger = logging.getLogger(__name__)

# Minimum seconds between consecutive liveness probe updates
LIVENESS_UPDATE_INTERVAL = 5.0


class RepoManager:
    """
    A client for downloading and syncing git repositories using GitPython.
    """

    repo_client: RepoClient
    git_repo: git.Repo | None
    tmp_dir: str
    repo_path: str
    initialization_future: Future | None

    _last_liveness_update: float
    _trigger_liveness_probe: Callable[[], None] | None
    _has_timed_out: bool

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
        self._has_timed_out = False
        self._last_liveness_update = 0.0

    @property
    def is_available(self):
        return self.git_repo is not None and self.initialization_future is None

    def initialize_in_background(self):
        logger.info(f"Creating initialize task for repo {self.repo_client.repo_full_name}")
        if self.initialization_future is not None:
            raise RuntimeError(
                f"Repo {self.repo_client.repo_full_name} is already being initialized"
            )

        self.initialization_future = ThreadPoolExecutor(1).submit(self.initialize)

    @sentry_sdk.trace
    def initialize(self):
        logger.info(f"Initializing repo {self.repo_client.repo_full_name}")

        try:
            self._clone_repo()
            if self._has_timed_out:
                return
            self._sync_repo()
        except Exception:
            logger.exception(
                "Failed to initialize repo", extra={"repo": self.repo_client.repo_full_name}
            )
            self.cleanup()
            raise
        finally:
            self.initialization_future = None

            if self._has_timed_out:
                self.cleanup()

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
            )
            end_time = time.time()
            logger.info(f"Cloned repository in {end_time - start_time} seconds")

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

            try:
                # First attempt with shallow fetch
                self.git_repo.git.execute(["git", "fetch", "--depth=1", "origin", commit_sha])
                self.git_repo.git.checkout(commit_sha)
            except git.GitCommandError as e:
                if "unable to read tree" in str(e.stderr):
                    # If we can't read the tree, try a full fetch of just the commit
                    logger.info(f"Shallow fetch failed for {commit_sha}, performing full fetch")
                    self.git_repo.git.execute(["git", "fetch", "--no-tags", "origin", commit_sha])
                    self.git_repo.git.checkout(commit_sha)
                else:
                    # If it's a different error, re-raise it
                    raise

            end_time = time.time()
            logger.info(
                f"Checked out repo {self.repo_client.repo_full_name} to commit {commit_sha} in {end_time - start_time} seconds"
            )
        except Exception:
            logger.exception(
                "Failed to sync repository", extra={"repo": self.repo_client.repo_full_name}
            )
            self.git_repo = None  # clear the repo to fail the available check

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

    def mark_as_timed_out(self):
        if self.initialization_future:
            # Have the thread deal with cleanup
            self._has_timed_out = True
        else:
            # Do it now
            self.cleanup()

    def cleanup(self):
        """
        Clean up the temporary directory if it exists.
        """
        if self.repo_path and os.path.exists(self.repo_path):
            try:
                cleanup_dir(self.repo_path)
            except Exception as e:
                logger.error(f"Error during repo cleanup, but continuing: {e}")
            finally:
                # Ensure we null out paths even if cleanup fails
                self.repo_path = None
                self.git_repo = None

        self._has_timed_out = False

        logger.info(f"Cleaned up repo for {self.repo_client.repo_full_name}")
