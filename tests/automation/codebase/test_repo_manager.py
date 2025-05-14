import datetime
import os
from concurrent.futures import Future
from unittest.mock import ANY, MagicMock, mock_open, patch

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


def test_clone_success(repo_manager, mock_repo_client):
    """Test successful repository cloning."""

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


def test_clone_failure_clears_repo(repo_manager, mock_repo_client):
    """Test that clone failure clears the repo reference."""

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


def test_sync_success(repo_manager, mock_repo_client):
    """Test successful repository sync."""

    # Setup mock git repo
    mock_git_repo = MagicMock(spec=git.Repo)
    repo_manager.git_repo = mock_git_repo

    repo_manager._sync_repo()

    # Verify git commands were called
    mock_git_repo.git.execute.assert_called_once_with(
        ["git", "fetch", "--depth=1", "origin", mock_repo_client.base_commit_sha]
    )
    mock_git_repo.git.checkout.assert_called_once_with(mock_repo_client.base_commit_sha, force=True)


def test_sync_failure_clears_repo(repo_manager, mock_repo_client):
    """Test that sync failure clears the repo reference."""

    # Setup mock git repo that raises on execute
    mock_git_repo = MagicMock(spec=git.Repo)
    mock_git_repo.git.execute.side_effect = Exception("Sync failed")
    repo_manager.git_repo = mock_git_repo

    repo_manager._sync_repo()

    # Verify repo was cleared
    assert repo_manager.git_repo is None


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


def test_initialize_in_background(
    repo_manager,
):
    """Test background initialization."""
    # First call should work
    repo_manager.initialize_in_background()
    assert repo_manager.initialization_future is not None

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


def test_gcs_archive_exists_blob_exists_raises_exception(repo_manager):
    """Test that if checking blob.exists raises, gcs_archive_exists logs and returns None."""
    dummy_entry = MagicMock()
    dummy_session = object()
    mock_blob = MagicMock()
    mock_blob.exists.side_effect = Exception("GCS error")
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    # Patch logger to assert exception log
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


def test_download_from_gcs_blob_not_exists(repo_manager, mock_repo_client):
    """Test that when the blob does not exist, download_from_gcs raises and logs."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    with (
        patch(
            "seer.automation.codebase.repo_manager.storage.Client",
            return_value=mock_storage_instance,
        ),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
    ):
        with pytest.raises(FileNotFoundError):
            repo_manager.download_from_gcs()


def test_download_from_gcs_success(repo_manager, mock_repo_client):
    """Test successful download_from_gcs sequence."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.size = 1024  # Add a numeric size for chunk calculations
    mock_blob.reload = MagicMock()  # Mock the reload method
    mock_blob.download_as_bytes = MagicMock(return_value=b"fake data")  # Mock download_as_bytes
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_instance = MagicMock()
    mock_storage_instance.bucket.return_value = mock_bucket

    fake_tar = MagicMock()
    fake_tar.__enter__.return_value = fake_tar
    fake_tar.getmembers.return_value = []
    fake_tar.extractall = MagicMock()

    with (
        patch(
            "seer.automation.codebase.repo_manager.storage.Client",
            return_value=mock_storage_instance,
        ),
        patch("seer.automation.codebase.repo_manager.cleanup_dir") as mock_cleanup_dir,
        patch("seer.automation.codebase.repo_manager.tarfile.open", return_value=fake_tar),
        patch("seer.automation.codebase.repo_manager.open", mock_open(), create=True),
        patch(
            "seer.automation.codebase.repo_manager.git.Repo", return_value=MagicMock(spec=git.Repo)
        ),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
        patch("seer.automation.codebase.repo_manager.os.makedirs"),
        patch("seer.automation.codebase.repo_manager.os.path.exists", return_value=True),
        patch("seer.automation.codebase.repo_manager.os.listdir", return_value=["some_file.txt"]),
        patch("seer.automation.codebase.repo_manager.os.unlink") as mock_unlink,
    ):
        repo_manager.download_from_gcs(chunk_size=512)  # Use smaller chunk size for test

        temp_tar = os.path.join(repo_manager.tmp_dir, "repo_archive.tar.gz")
        mock_cleanup_dir.assert_called_once_with(repo_manager.repo_path)
        mock_storage_instance.bucket.assert_called_once_with(repo_manager.get_bucket_name())
        mock_bucket.blob.assert_called_once_with(repo_manager.blob_name)
        mock_blob.exists.assert_called_once()
        mock_blob.reload.assert_called_once()  # Verify reload was called

        fake_tar.getmembers.assert_called_once()
        fake_tar.extractall.assert_called_once_with(path=repo_manager.repo_path, members=[])
        mock_unlink.assert_called_once_with(temp_tar)
        assert repo_manager.git_repo is not None


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


def test_upload_lock_already_locked(repo_manager):
    """Test that upload_lock raises when the archive is already locked."""
    now = datetime.datetime.now(datetime.timezone.utc)
    dummy_archive = MagicMock()
    dummy_archive.upload_locked_at = now
    dummy_session = MagicMock()
    with (
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=dummy_archive),
    ):
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = dummy_session

        with pytest.raises(RepoInitializationError):
            with repo_manager.upload_lock():
                pass

        assert dummy_archive.upload_locked_at == now


def test_upload_to_gcs_success(repo_manager, mock_repo_client):
    """Test upload_to_gcs uploads tar to GCS and cleans up temp files."""
    # Prepare a dummy archive record for upload_lock
    dummy_archive = MagicMock()
    dummy_archive.upload_locked_at = None
    copied_path = os.path.join(repo_manager.tmp_dir, "repo")
    with (
        patch.object(repo_manager, "_copy_repo", return_value=copied_path),
        patch.object(repo_manager, "_verify_repo_state"),
        patch.object(repo_manager, "_prune_repo"),
        patch("seer.automation.codebase.repo_manager.os.listdir", return_value=["foo"]),
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(
            repo_manager, "get_db_archive_entry", side_effect=[None, dummy_archive, dummy_archive]
        ),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch("seer.automation.codebase.repo_manager.tarfile.open") as mock_tar_open,
        patch("seer.automation.codebase.repo_manager.os.path.exists", return_value=True),
        patch("seer.automation.codebase.repo_manager.os.unlink") as mock_unlink,
        patch("seer.automation.codebase.repo_manager.shutil.rmtree") as mock_rmtree,
    ):
        # Setup session
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = mock_session
        mock_session.commit = MagicMock()
        mock_session.add = MagicMock()

        # Setup tar
        fake_tar = MagicMock()
        mock_tar_open.return_value.__enter__.return_value = fake_tar

        # Setup storage
        storage_instance = MagicMock()
        bucket = MagicMock()
        blob = MagicMock()
        storage_instance.bucket.return_value = bucket
        bucket.blob.return_value = blob
        mock_storage_client.return_value = storage_instance

        # Run
        repo_manager.upload_to_gcs()

        # DB actions should have added a new archive and committed
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called()

        # Tar and upload calls
        temp_tar = os.path.join(repo_manager.tmp_dir, "upload_repo_archive.tar.gz")
        mock_tar_open.assert_called_once_with(temp_tar, "w:gz")
        fake_tar.add.assert_called_once_with(os.path.join(copied_path, "foo"), arcname="foo")
        storage_instance.bucket.assert_called_once_with("bucket-name")
        bucket.blob.assert_called_once_with(repo_manager.blob_name)
        blob.upload_from_filename.assert_called_once_with(temp_tar)

        # Cleanup
        mock_unlink.assert_called_once_with(temp_tar)
        mock_rmtree.assert_called_once_with(copied_path)


def test_upload_to_gcs_failure(repo_manager, mock_repo_client):
    """Test upload_to_gcs logs and propagates exception during upload."""
    # Prepare a dummy archive record for upload_lock
    dummy_archive = MagicMock()
    dummy_archive.upload_locked_at = None
    copied_path = os.path.join(repo_manager.tmp_dir, "repo")
    with (
        patch.object(repo_manager, "_copy_repo", return_value=copied_path),
        patch.object(repo_manager, "_verify_repo_state"),
        patch.object(repo_manager, "_prune_repo"),
        patch("seer.automation.codebase.repo_manager.os.listdir", return_value=["bar"]),
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_class,
        patch.object(repo_manager, "get_db_archive_entry", return_value=dummy_archive),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch("seer.automation.codebase.repo_manager.tarfile.open") as mock_tar_open,
        patch("seer.automation.codebase.repo_manager.os.path.exists", return_value=True),
        patch("seer.automation.codebase.repo_manager.os.unlink") as mock_unlink,
        patch("seer.automation.codebase.repo_manager.shutil.rmtree") as mock_rmtree,
    ):
        # Setup session
        mock_session = mock_session_class.return_value
        mock_session.__enter__.return_value = mock_session
        mock_session.commit = MagicMock()
        mock_session.add = MagicMock()

        # Setup tar
        fake_tar = MagicMock()
        mock_tar_open.return_value.__enter__.return_value = fake_tar

        # Setup storage that raises
        storage_instance = MagicMock()
        bucket = MagicMock()
        blob = MagicMock()
        blob.upload_from_filename.side_effect = Exception("upload error")
        storage_instance.bucket.return_value = bucket
        bucket.blob.return_value = blob
        mock_storage_client.return_value = storage_instance

        # Run and assert exception
        with pytest.raises(Exception, match="upload error"):
            repo_manager.upload_to_gcs()

        # Cleanup still executed
        temp_tar = os.path.join(repo_manager.tmp_dir, "upload_repo_archive.tar.gz")
        mock_unlink.assert_called_once_with(temp_tar)
        mock_rmtree.assert_called_once_with(copied_path)


def test_download_from_gcs_prevent_path_traversal(repo_manager, mock_repo_client):
    """Test that download_from_gcs skips unsafe paths to prevent path traversal and completes successfully."""
    # Setup storage blob
    storage_instance = MagicMock()
    bucket = MagicMock()
    blob = MagicMock()
    blob.exists.return_value = True
    blob.size = 1024  # Add a numeric size for chunk calculations
    blob.reload = MagicMock()  # Mock the reload method
    blob.download_as_bytes = MagicMock(return_value=b"fake data")  # Mock download_as_bytes
    storage_instance.bucket.return_value = bucket
    bucket.blob.return_value = blob

    # Create fake tar members
    fake_member_safe = MagicMock()
    fake_member_safe.name = "copied_repo/safe.txt"
    fake_member_unsafe = MagicMock()
    fake_member_unsafe.name = "copied_repo/../unsafe.txt"
    fake_member_abs = MagicMock()
    fake_member_abs.name = "/absolute/path"

    fake_tar = MagicMock()

    with (
        patch(
            "seer.automation.codebase.repo_manager.storage.Client", return_value=storage_instance
        ),
        patch.object(repo_manager, "get_bucket_name", return_value="bucket-name"),
        patch("seer.automation.codebase.repo_manager.cleanup_dir"),
        patch("seer.automation.codebase.repo_manager.os.makedirs"),
        patch("seer.automation.codebase.repo_manager.os.listdir", return_value=["safe.txt"]),
        patch("seer.automation.codebase.repo_manager.os.path.exists", return_value=True),
        patch("seer.automation.codebase.repo_manager.open", mock_open(), create=True),
        patch("seer.automation.codebase.repo_manager.tarfile.open") as mock_tar_open,
        patch(
            "seer.automation.codebase.repo_manager.git.Repo", return_value=MagicMock(spec=git.Repo)
        ),
        patch("seer.automation.codebase.repo_manager.os.unlink"),
    ):
        # Configure tarfile context manager
        mock_tar_open.return_value.__enter__.return_value = fake_tar
        fake_tar.getmembers.return_value = [fake_member_safe, fake_member_unsafe, fake_member_abs]
        fake_tar.extractall = MagicMock()

        # Run download
        repo_manager.download_from_gcs(chunk_size=512)  # Use smaller chunk size for test

        # Member name should be sanitized for safe member
        assert fake_member_safe.name == "safe.txt"

        # extractall called with only the safe member
        args, kwargs = fake_tar.extractall.call_args
        assert kwargs["path"] == repo_manager.repo_path
        members_list = kwargs.get("members", [])
        assert fake_member_safe in members_list
        assert fake_member_unsafe not in members_list
        assert fake_member_abs not in members_list

        # Git repo should be set
        assert repo_manager.git_repo is not None


def test_prune_repo_not_initialized(repo_manager):
    """Test that _prune_repo raises when repository is not initialized."""
    mock_git = MagicMock()
    mock_git.git = MagicMock()
    # status returns empty -> not initialized
    mock_git.git.execute.return_value = ""
    with patch("seer.automation.codebase.repo_manager.git.Repo", return_value=mock_git):
        with pytest.raises(RepoInitializationError, match="Repository is not initialized"):
            repo_manager._prune_repo()


def test_prune_repo_no_refs(repo_manager):
    """Test that _prune_repo skips deletion when no refs are found."""
    mock_git = MagicMock()

    def fake_execute(*args, **kwargs):
        cmd = kwargs.get("command", args[0] if args else None)
        if cmd == ["git", "status"]:
            return "status"
        if args and args[0] == ["git", "show-ref"]:
            return ""
        return None

    mock_git.git = MagicMock()
    mock_git.git.execute.side_effect = fake_execute
    with patch("seer.automation.codebase.repo_manager.git.Repo", return_value=mock_git):
        repo_manager._prune_repo()
        # Ensure update-ref not called
        for args, kwargs in mock_git.git.execute.call_args_list:
            if args and isinstance(args[0], list) and args[0][1] == "update-ref":
                pytest.fail("update-ref should not be called when no refs")


def test_prune_repo_delete_refs(repo_manager):
    """Test that _prune_repo deletes refs that do not include base_commit_sha."""
    base_sha = repo_manager.repo_client.base_commit_sha
    show_refs = f"{base_sha} refs/heads/keep\notherref refs/heads/delete1"
    mock_git = MagicMock()
    side_effects = [
        "status",  # status
        show_refs,  # show-ref
        None,  # update-ref delete1
        None,  # reflog expire
        None,  # gc
    ]
    mock_git.git = MagicMock()
    mock_git.git.execute.side_effect = side_effects
    with patch("seer.automation.codebase.repo_manager.git.Repo", return_value=mock_git):
        repo_manager._prune_repo()
        # Ensure update-ref called for delete1
        mock_git.git.execute.assert_any_call(["git", "update-ref", "-d", "refs/heads/delete1"])


def test_prune_repo_show_ref_error(repo_manager):
    """Test that _prune_repo handles show-ref errors gracefully."""
    mock_git = MagicMock()
    error = git.GitCommandError("git show-ref", 1)
    side_effects = [
        "status",  # status
        error,  # show-ref raises
        None,  # reflog expire
        None,  # gc
    ]
    mock_git.git = MagicMock()
    mock_git.git.execute.side_effect = side_effects
    with patch("seer.automation.codebase.repo_manager.git.Repo", return_value=mock_git):
        repo_manager._prune_repo()
        mock_git.git.execute.assert_any_call(["git", "reflog", "expire", "--expire=now", "--all"])
        mock_git.git.execute.assert_any_call(["git", "gc", "--prune=now", "--aggressive"])
