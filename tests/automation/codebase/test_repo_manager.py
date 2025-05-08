import logging
from concurrent.futures import Future
from unittest.mock import ANY, MagicMock, patch

import git
import pytest

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.repo_manager import RepoManager


@pytest.fixture
def mock_repo_client():
    client = MagicMock(spec=RepoClient)
    client.repo_owner = "test-owner"
    client.repo_name = "test-repo"
    client.base_commit_sha = "abcd123"
    client.repo_full_name = "test-owner/test-repo"
    return client


@pytest.fixture
def repo_manager(mock_repo_client, tmp_path):
    """Create a RepoManager instance with a mock repo client."""
    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
        manager = RepoManager(repo_client=mock_repo_client)
        yield manager


def test_clone_success(repo_manager, mock_repo_client, caplog):
    """Test successful repository cloning and checkout of a specific commit."""
    caplog.set_level(logging.INFO)

    # Setup mock repo client to return a local file URL and a fake commit sha
    mock_repo_client.get_clone_url_with_auth.return_value = "file:///fake/repo.git"
    mock_repo_client.base_commit_sha = "deadbeef"

    # Create a mock git.Repo object with mock git attribute
    mock_git = MagicMock()
    mock_git_repo = MagicMock(spec=git.Repo)
    mock_git_repo.git = mock_git

    with (
        patch("git.Repo.clone_from", return_value=mock_git_repo) as mock_clone,
        patch("seer.automation.codebase.repo_manager.cleanup_dir"),
    ):
        repo_manager._clone_repo()

        # Verify clone was called with correct arguments, using ANY for the progress callback
        mock_clone.assert_called_once_with(
            "file:///fake/repo.git",
            repo_manager.repo_path,
            depth=1,
            progress=ANY,
        )

        # Verify fetch and checkout were called with correct arguments
        mock_git.fetch.assert_called_once_with("origin", "deadbeef", depth=1)
        mock_git.checkout.assert_called_once_with("deadbeef")

        # Verify the repo was set
        assert repo_manager.git_repo == mock_git_repo

        # Check logging
        assert "Cloning repository test-owner/test-repo" in caplog.text
        assert "Cloned and checked out repository at commit deadbeef" in caplog.text


def test_clone_failure_clears_repo(repo_manager, mock_repo_client, caplog):
    """Test that clone failure clears the repo reference."""
    caplog.set_level(logging.ERROR)

    mock_repo_client.get_clone_url_with_auth.return_value = "file:///fake/repo.git"

    # Make clone_from raise GitCommandError
    error = git.GitCommandError("git clone", 128)
    with (
        patch("git.Repo.clone_from", side_effect=error),
        patch("seer.automation.codebase.repo_manager.cleanup_dir"),
    ):
        with pytest.raises(git.GitCommandError):
            repo_manager._clone_repo()

        # Verify repo was cleared
        assert repo_manager.git_repo is None

        # Verify error was logged
        assert "Failed to clone repository" in caplog.text


def test_sync_failure_clears_repo(repo_manager, mock_repo_client, caplog):
    """Test that sync failure clears the repo reference."""
    caplog.set_level(logging.ERROR)

    # Setup mock git repo that raises on execute
    mock_git_repo = MagicMock(spec=git.Repo)
    mock_git_repo.git.execute.side_effect = Exception("Sync failed")
    repo_manager.git_repo = mock_git_repo

    repo_manager._sync_repo()

    # Verify repo was cleared
    assert repo_manager.git_repo is None

    # Verify error was logged
    assert f"Failed to sync repository {mock_repo_client.repo_full_name}" in caplog.text


def test_mark_as_timed_out_before_init(repo_manager):
    """Test marking as timed out before initialization."""
    repo_path = repo_manager.repo_path

    with (
        patch("seer.automation.codebase.repo_manager.cleanup_dir") as mock_cleanup,
        patch("os.path.exists", return_value=True),
    ):
        repo_manager.mark_as_timed_out()

        # Verify cleanup was called
        mock_cleanup.assert_called_once_with(repo_path)
        assert repo_manager.git_repo is None
        assert not repo_manager._has_timed_out


def test_mark_as_timed_out_during_init(repo_manager):
    """Test marking as timed out during initialization."""
    repo_manager.initialization_future = Future()

    with patch("seer.automation.codebase.repo_manager.cleanup_dir"):
        repo_manager.mark_as_timed_out()

        # Verify cleanup was deferred
        assert repo_manager._has_timed_out
        assert repo_manager.repo_path is not None


def test_cleanup_idempotent(repo_manager):
    """Test that cleanup can be called multiple times safely."""
    repo_path = repo_manager.repo_path

    with (
        patch("seer.automation.codebase.repo_manager.cleanup_dir") as mock_cleanup,
        patch("os.path.exists", return_value=True),
    ):
        # First cleanup
        repo_manager.cleanup()
        mock_cleanup.assert_called_once_with(repo_path)

        # Reset the mock for second call
        mock_cleanup.reset_mock()

        # Second cleanup should not call cleanup_dir again since path is None
        repo_manager.cleanup()
        mock_cleanup.assert_not_called()


def test_is_available_states(repo_manager):
    """Test different states of the is_available property."""
    # Initially should be False (no git_repo)
    assert not repo_manager.is_available

    # Set git_repo but with pending initialization
    repo_manager.git_repo = MagicMock()
    repo_manager.initialization_future = Future()
    assert not repo_manager.is_available

    # Clear initialization future
    repo_manager.initialization_future = None
    assert repo_manager.is_available

    # Clear git_repo
    repo_manager.git_repo = None
    assert not repo_manager.is_available


def test_initialize_in_background(repo_manager, caplog):
    """Test background initialization."""
    caplog.set_level(logging.INFO)

    # First call should work
    repo_manager.initialize_in_background()
    assert repo_manager.initialization_future is not None
    assert (
        f"Creating initialize task for repo {repo_manager.repo_client.repo_full_name}"
        in caplog.text
    )

    # Second call should raise
    with pytest.raises(RuntimeError, match="is already being initialized"):
        repo_manager.initialize_in_background()


def test_initialize_success(repo_manager, mock_repo_client):
    """Test successful initialization sequence."""
    with (
        patch.object(repo_manager, "_clone_repo") as mock_clone,
        patch.object(repo_manager, "_sync_repo") as mock_sync,
    ):
        repo_manager.initialize()

        # Verify sequence
        mock_clone.assert_called_once()
        mock_sync.assert_called_once()
        assert repo_manager.initialization_future is None


def test_initialize_cleans_up_on_timeout(repo_manager):
    """Test that initialization cleans up when timed out."""
    repo_manager._has_timed_out = True

    with (
        patch.object(repo_manager, "_clone_repo") as mock_clone,
        patch.object(repo_manager, "cleanup") as mock_cleanup,
    ):
        repo_manager.initialize()

        mock_clone.assert_called_once()
        mock_cleanup.assert_called_once()
