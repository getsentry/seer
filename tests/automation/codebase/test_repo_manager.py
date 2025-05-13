import datetime
import logging
import os
from concurrent.futures import Future
from unittest.mock import ANY, MagicMock, patch

import git
import pytest

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.repo_manager import RepoInitializationError, RepoManager


@pytest.fixture
def mock_repo_client():
    client = MagicMock(spec=RepoClient)
    client.repo_owner = "test-owner"
    client.repo_name = "test-repo"
    client.base_commit_sha = "abcd123"
    client.repo_full_name = "test-owner/test-repo"
    client.provider = "github"
    client.repo_external_id = "1234567890"
    client.get_current_commit_info.return_value = {
        "timestamp": "2021-01-01",
    }
    return client


@pytest.fixture
def repo_manager(mock_repo_client, tmp_path):
    """Create a RepoManager instance with a mock repo client."""
    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
        manager = RepoManager(repo_client=mock_repo_client, organization_id=1, project_id=6178942)
        yield manager


def test_clone_success(repo_manager, mock_repo_client, caplog):
    """Test successful repository cloning."""
    caplog.set_level(logging.INFO)

    # Setup mock repo client to return a local file URL
    mock_repo_client.get_clone_url_with_auth.return_value = "file:///fake/repo.git"

    # Create a mock git.Repo object
    mock_git_repo = MagicMock(spec=git.Repo)

    with (
        patch("git.Repo.clone_from", return_value=mock_git_repo) as mock_clone,
        patch("seer.automation.codebase.repo_manager.cleanup_dir"),
    ):

        repo_manager._clone_repo()

        # Verify clone was called with correct arguments, using ANY for the progress callback
        mock_clone.assert_called_once_with(
            "file:///fake/repo.git",
            repo_manager.repo_path,
            progress=ANY,  # Use ANY to match any function
            depth=1,
        )

        # Verify the repo was set
        assert repo_manager.git_repo == mock_git_repo

        # Check logging
        assert "Cloning repository test-owner/test-repo" in caplog.text
        assert "Cloned repository" in caplog.text


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


def test_sync_success(repo_manager, mock_repo_client, caplog):
    """Test successful repository sync."""
    caplog.set_level(logging.INFO)

    # Setup mock git repo
    mock_git_repo = MagicMock(spec=git.Repo)
    repo_manager.git_repo = mock_git_repo

    repo_manager._sync_repo()

    # Verify git commands were called
    mock_git_repo.git.execute.assert_called_once_with(
        ["git", "fetch", "--depth=1", "origin", mock_repo_client.base_commit_sha]
    )
    mock_git_repo.git.checkout.assert_called_once_with(mock_repo_client.base_commit_sha, force=True)

    # Check logging
    assert f"Syncing repository {mock_repo_client.repo_full_name}" in caplog.text
    assert "Checked out repo" in caplog.text


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
    assert "Failed to sync repository" in caplog.text


def test_mark_as_timed_out_before_init(repo_manager):
    """Test marking as timed out before initialization."""
    tmp_dir = repo_manager.tmp_dir

    with (
        patch("seer.automation.codebase.repo_manager.cleanup_dir") as mock_cleanup,
        patch("os.path.exists", return_value=True),
    ):
        repo_manager.mark_as_timed_out()

        # Verify cleanup was called
        mock_cleanup.assert_called_once_with(tmp_dir)
        assert repo_manager.git_repo is None
        assert repo_manager.is_cancelled


def test_mark_as_timed_out_during_init(repo_manager):
    """Test marking as timed out during initialization."""
    repo_manager.initialization_future = Future()

    with patch("seer.automation.codebase.repo_manager.cleanup_dir"):
        repo_manager.mark_as_timed_out()

        # Verify cleanup was deferred
        assert repo_manager.is_cancelled
        assert repo_manager.repo_path is not None


def test_cleanup_idempotent(repo_manager):
    """Test that cleanup can be called multiple times safely."""
    tmp_dir = repo_manager.tmp_dir

    with (
        patch("seer.automation.codebase.repo_manager.cleanup_dir") as mock_cleanup,
        patch("os.path.exists", return_value=True),
    ):
        # First cleanup
        repo_manager.cleanup()
        mock_cleanup.assert_called_once_with(tmp_dir)

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


def test_initialize_success_without_gcs(repo_manager, mock_repo_client):
    """Test successful initialization sequence."""
    repo_manager._use_gcs = False

    with (
        patch.object(repo_manager, "_clone_repo") as mock_clone,
        patch.object(repo_manager, "download_from_gcs") as mock_download,
        patch.object(repo_manager, "_sync_repo") as mock_sync,
        patch.object(repo_manager, "_copy_repo", return_value="copied_repo_path") as mock_copy,
        patch.object(repo_manager, "upload_to_gcs") as mock_upload,
    ):
        repo_manager.initialize()

        # Verify sequence
        mock_clone.assert_called_once()
        mock_download.assert_not_called()
        mock_sync.assert_called_once()
        mock_copy.assert_not_called()
        mock_upload.assert_not_called()
        assert repo_manager.initialization_future is None


def test_initialize_from_gcs_download(repo_manager, mock_repo_client):
    """Test initialization from a GCS download."""
    repo_manager._use_gcs = True

    mock_repo_client.get_current_commit_info.side_effect = [
        {"timestamp": "2021-01-01"},
        {"timestamp": "2021-01-01"},
    ]

    with (
        patch.object(repo_manager, "_clone_repo") as mock_clone,
        patch.object(
            repo_manager, "gcs_archive_exists", return_value=MagicMock(commit_sha="123")
        ) as mock_gcs_exists,
        patch.object(repo_manager, "download_from_gcs") as mock_download,
        patch.object(repo_manager, "_sync_repo") as mock_sync,
        patch.object(repo_manager, "_copy_repo", return_value="copied_repo_path") as mock_copy,
        patch.object(repo_manager, "upload_to_gcs") as mock_upload,
    ):
        repo_manager.initialize()

        # Verify sequence
        mock_clone.assert_not_called()
        mock_gcs_exists.assert_called_once()
        mock_download.assert_called_once()
        mock_sync.assert_called_once()
        mock_copy.assert_not_called()
        mock_upload.assert_not_called()
        assert repo_manager.initialization_future is None


def test_initialize_cleans_up_on_timeout(repo_manager):
    """Test that initialization cleans up when timed out."""
    repo_manager.is_cancelled = True

    with (
        patch.object(repo_manager, "_clone_repo") as mock_clone,
        patch.object(repo_manager, "cleanup") as mock_cleanup,
    ):
        repo_manager.initialize()

        mock_clone.assert_not_called()
        mock_cleanup.assert_called_once()


def test_gcs_archive_exists_no_db_entry(repo_manager):
    """Test that when no DB entry exists, gcs_archive_exists returns None without calling GCS."""
    dummy_session = object()
    with (
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=None) as mock_get_db,
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
    ):
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = dummy_session

        result = repo_manager.gcs_archive_exists()

        assert result is None
        mock_get_db.assert_called_once_with(dummy_session)
        mock_storage_client.assert_not_called()


def test_gcs_archive_exists_db_entry_blob_not_exists(repo_manager):
    """Test that when a DB entry exists but the blob does not exist in GCS, returns None."""
    dummy_entry = MagicMock()
    dummy_session = object()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    with (
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=dummy_entry) as mock_get_db,
        patch(
            "seer.automation.codebase.repo_manager.storage.Client",
            return_value=mock_storage_instance,
        ) as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
    ):
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = dummy_session

        result = repo_manager.gcs_archive_exists()

        assert result is None
        mock_get_db.assert_called_once_with(dummy_session)
        mock_storage_client.assert_called_once()
        mock_storage_instance.bucket.assert_called_once_with("bucket-name")
        mock_bucket.blob.assert_called_once_with(repo_manager.blob_name)
        mock_blob.exists.assert_called_once()


def test_gcs_archive_exists_db_entry_blob_exists(repo_manager):
    """Test that when a DB entry exists and the blob exists in GCS, returns the entry."""
    dummy_entry = MagicMock()
    dummy_session = object()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    with (
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=dummy_entry) as mock_get_db,
        patch(
            "seer.automation.codebase.repo_manager.storage.Client",
            return_value=mock_storage_instance,
        ) as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
    ):
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = dummy_session

        result = repo_manager.gcs_archive_exists()

        assert result == dummy_entry
        assert mock_get_db.call_count == 2
        mock_storage_client.assert_called_once()
        mock_storage_instance.bucket.assert_called_once_with("bucket-name")
        mock_bucket.blob.assert_called_once_with(repo_manager.blob_name)
        mock_blob.exists.assert_called_once()


def test_gcs_archive_exists_blob_exists_raises_exception(repo_manager, caplog):
    """Test that if checking blob.exists raises, gcs_archive_exists logs and returns None."""
    dummy_entry = MagicMock()
    dummy_session = object()
    mock_blob = MagicMock()
    mock_blob.exists.side_effect = Exception("GCS error")
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    caplog.set_level(logging.ERROR)
    with (
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=dummy_entry),
        patch(
            "seer.automation.codebase.repo_manager.storage.Client",
            return_value=mock_storage_instance,
        ),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
    ):
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = dummy_session

        result = repo_manager.gcs_archive_exists()

        assert result is None
        assert "Error checking if repository archive exists in GCS" in caplog.text


def test_download_from_gcs_blob_not_exists(repo_manager, mock_repo_client, caplog):
    """Test that when the blob does not exist, download_from_gcs raises and logs."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    caplog.set_level(logging.ERROR)
    with (
        patch(
            "seer.automation.codebase.repo_manager.storage.Client",
            return_value=mock_storage_instance,
        ),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
    ):
        with pytest.raises(FileNotFoundError):
            repo_manager.download_from_gcs()
    assert "Failed to download repository from GCS" in caplog.text


def test_download_from_gcs_success(repo_manager, mock_repo_client, caplog):
    """Test successful download_from_gcs sequence."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_to_filename = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    fake_tar = MagicMock()
    fake_tar.__enter__.return_value = fake_tar
    fake_tar.getmembers.return_value = []
    fake_tar.extractall = MagicMock()

    caplog.set_level(logging.INFO)
    with (
        patch(
            "seer.automation.codebase.repo_manager.storage.Client",
            return_value=mock_storage_instance,
        ),
        patch("seer.automation.codebase.repo_manager.cleanup_dir") as mock_cleanup_dir,
        patch("seer.automation.codebase.repo_manager.tarfile.open", return_value=fake_tar),
        patch(
            "seer.automation.codebase.repo_manager.git.Repo", return_value=MagicMock(spec=git.Repo)
        ),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
    ):
        repo_manager.download_from_gcs()

        temp_tar = os.path.join(repo_manager.tmp_dir, "repo_archive.tar.gz")
        mock_cleanup_dir.assert_called_once_with(repo_manager.repo_path)
        mock_storage_instance.bucket.assert_called_once_with(repo_manager.get_bucket_name())
        mock_bucket.blob.assert_called_once_with(repo_manager.blob_name)
        mock_blob.exists.assert_called_once()
        mock_blob.download_to_filename.assert_called_once_with(temp_tar)
        fake_tar.getmembers.assert_called_once()
        fake_tar.extractall.assert_called_once_with(path=repo_manager.repo_path, members=[])
        assert repo_manager.git_repo is not None
        assert "Successfully downloaded repository from GCS" in caplog.text


def test_upload_lock_success(repo_manager):
    """Test that upload_lock sets and clears the lock when not already locked."""
    dummy_archive = MagicMock()
    dummy_archive.upload_locked_at = None
    dummy_session = MagicMock()
    with (
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=dummy_archive),
    ):
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = dummy_session

        # Acquire and release the lock
        with repo_manager.upload_lock():
            # Lock should be set
            assert dummy_archive.upload_locked_at is not None
            # Commit should be called once for locking
            assert dummy_session.commit.call_count == 1

        # After context, lock should be cleared
        assert dummy_archive.upload_locked_at is None
        # Commit should be called again for clearing
        assert dummy_session.commit.call_count == 2


def test_upload_lock_already_locked(repo_manager, caplog):
    """Test that upload_lock raises when the archive is already locked."""
    now = datetime.datetime.now(datetime.timezone.utc)
    dummy_archive = MagicMock()
    dummy_archive.upload_locked_at = now
    dummy_session = MagicMock()
    caplog.set_level(logging.INFO)
    with (
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=dummy_archive),
    ):
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = dummy_session

        with pytest.raises(RepoInitializationError):
            with repo_manager.upload_lock():
                pass

    assert "Repository is already locked for upload" in caplog.text
