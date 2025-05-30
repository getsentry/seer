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

    # Add repo and github_auth attributes for tarball download functionality
    mock_github_repo = MagicMock()
    mock_github_repo.get_archive_link.return_value = (
        "https://api.github.com/repos/test-owner/test-repo/tarball/abcd123"
    )
    client.repo = mock_github_repo

    mock_github_auth = MagicMock()
    mock_github_auth.token = "test-token"
    client.github_auth = mock_github_auth

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
            no_checkout=True,
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

    mock_repo_client.get_clone_url_with_auth.return_value = (
        "https://auth-token@github.com/test-owner/test-repo.git"
    )

    repo_manager._sync_repo()

    # Verify git commands were called
    mock_git_repo.git.execute.assert_called_once_with(
        [
            "git",
            "fetch",
            "--depth=1",
            "https://auth-token@github.com/test-owner/test-repo.git",
            mock_repo_client.base_commit_sha,
        ]
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
    """Test successful initialization sequence using tarball download."""
    with (patch.object(repo_manager, "_download_github_tar") as mock_download_tar,):
        repo_manager.initialize()

        # Verify tarball download was called instead of clone/sync
        mock_download_tar.assert_called_once()
        assert repo_manager.initialization_future is None


def test_initialize_from_gcs_download(repo_manager, mock_repo_client):
    """Test initialization with tarball download (previously GCS download test)."""
    with (patch.object(repo_manager, "_download_github_tar") as mock_download_tar,):
        repo_manager.initialize()

        # Verify tarball download was called
        mock_download_tar.assert_called_once()
        assert repo_manager.initialization_future is None


def test_initialize_tarball_download_success(repo_manager, mock_repo_client):
    """Test successful initialization using GitHub tarball download."""
    # Mock successful tarball extraction
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

    fake_tar = MagicMock()
    fake_tar.__enter__.return_value = fake_tar
    fake_tar.getmembers.return_value = [MagicMock(name="file1.py")]

    with (
        patch("seer.automation.codebase.repo_manager.requests.get", return_value=mock_response),
        patch("seer.automation.codebase.repo_manager.tarfile.open", return_value=fake_tar),
        patch("seer.automation.codebase.repo_manager.cleanup_dir"),
        patch("seer.automation.codebase.repo_manager.os.makedirs"),
        patch(
            "seer.automation.codebase.repo_manager.os.listdir", return_value=["extracted_folder"]
        ),
        patch("seer.automation.codebase.repo_manager.os.path.isdir", return_value=True),
        patch("seer.automation.codebase.repo_manager.shutil.move"),
        patch("seer.automation.codebase.repo_manager.shutil.rmtree"),
        patch("seer.automation.codebase.repo_manager.os.path.exists", return_value=True),
        patch("seer.automation.codebase.repo_manager.os.unlink"),
        patch("seer.automation.codebase.repo_manager.git.Repo.init") as mock_repo_init,
        patch("seer.automation.codebase.repo_manager.open", mock_open(), create=True),
        patch.object(repo_manager, "_verify_repo_state"),
    ):
        mock_git_repo = MagicMock(spec=git.Repo)
        mock_repo_init.return_value = mock_git_repo

        repo_manager.initialize()

        # Verify the process completed successfully
        mock_repo_client.repo.get_archive_link.assert_called_once_with("tarball", ref="abcd123")
        mock_repo_init.assert_called_once_with(repo_manager.repo_path)
        assert repo_manager.git_repo == mock_git_repo
        assert repo_manager.initialization_future is None


def test_initialize_cleans_up_on_timeout(repo_manager):
    """Test that initialization cleans up when timed out."""
    repo_manager.is_cancelled = True

    with (
        patch.object(repo_manager, "_download_github_tar") as mock_download_tar,
        patch.object(repo_manager, "cleanup") as mock_cleanup,
    ):
        repo_manager.initialize()

        mock_download_tar.assert_called_once()
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
    # Create a test database record that simulates an existing archive
    from seer.automation.models import RepoDefinition
    from seer.db import DbSeerRepoArchive, Session

    repo_definition = RepoDefinition(
        provider="github",
        owner="test-owner",
        name="test-repo",
        external_id="1234567890",
    )

    # Create initial archive without last_downloaded_at
    with Session() as session:
        repo_archive = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="bucket-name",
            blob_path=repo_manager.blob_name,
            commit_sha="abcd123",
            repo_definition=repo_definition.model_dump(),
            last_downloaded_at=None,  # Initially not downloaded
        )
        session.add(repo_archive)
        session.commit()
        initial_archive_id = repo_archive.id

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

        # Verify that last_downloaded_at was updated
        with Session() as session:
            updated_archive = (
                session.query(DbSeerRepoArchive)
                .filter(DbSeerRepoArchive.id == initial_archive_id)
                .first()
            )
            assert updated_archive is not None
            assert updated_archive.last_downloaded_at is not None
            # Verify it's a recent timestamp (within the last minute)
            import datetime

            time_diff = datetime.datetime.now(
                datetime.UTC
            ) - updated_archive.last_downloaded_at.replace(tzinfo=datetime.UTC)
            assert time_diff.total_seconds() < 60


def test_download_from_gcs_success_no_db_entry(repo_manager, mock_repo_client):
    """Test successful download_from_gcs when no database entry exists."""
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
        # Should complete successfully even without a database entry
        repo_manager.download_from_gcs(chunk_size=512)

        temp_tar = os.path.join(repo_manager.tmp_dir, "repo_archive.tar.gz")
        mock_cleanup_dir.assert_called_once_with(repo_manager.repo_path)
        mock_storage_instance.bucket.assert_called_once_with(repo_manager.get_bucket_name())
        mock_bucket.blob.assert_called_once_with(repo_manager.blob_name)
        mock_blob.exists.assert_called_once()
        mock_blob.reload.assert_called_once()

        fake_tar.getmembers.assert_called_once()
        fake_tar.extractall.assert_called_once_with(path=repo_manager.repo_path, members=[])
        mock_unlink.assert_called_once_with(temp_tar)
        assert repo_manager.git_repo is not None

        # Verify no database entries were created (since this is just a download without existing entry)
        from seer.db import DbSeerRepoArchive, Session

        with Session() as session:
            archives = session.query(DbSeerRepoArchive).all()
            assert len(archives) == 0


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
    copied_path = os.path.join(repo_manager.tmp_dir, "copied_repo")
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
    copied_path = os.path.join(repo_manager.tmp_dir, "copied_repo")
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
        None,  # remote remove
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
        None,  # remote remove
    ]
    mock_git.git = MagicMock()
    mock_git.git.execute.side_effect = side_effects
    with patch("seer.automation.codebase.repo_manager.git.Repo", return_value=mock_git):
        repo_manager._prune_repo()
        mock_git.git.execute.assert_any_call(["git", "reflog", "expire", "--expire=now", "--all"])
        mock_git.git.execute.assert_any_call(["git", "gc", "--prune=now"])


def test_delete_archive_success(repo_manager):
    """Test successful deletion of both GCS blob and database record."""
    from seer.automation.models import RepoDefinition
    from seer.db import DbSeerRepoArchive, Session

    # Create a test database record
    repo_definition = RepoDefinition(
        provider="github",
        owner="test-owner",
        name="test-repo",
        external_id="1234567890",
    )

    with Session() as session:
        repo_archive = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="test-bucket",
            blob_path="repos/1/github/test-owner/test-repo_1234567890.tar.gz",
            commit_sha="abcd123",
            repo_definition=repo_definition.model_dump(),
        )
        session.add(repo_archive)
        session.commit()

    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
    ):
        # Mock GCS blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Execute delete_archive
        repo_manager.delete_archive()

        # Verify GCS blob deletion
        mock_storage_client.assert_called_once()
        mock_bucket.blob.assert_called_once_with(repo_manager.blob_name)
        mock_blob.exists.assert_called_once()
        mock_blob.delete.assert_called_once()

        # Verify database record is deleted
        with Session() as session:
            remaining_archives = (
                session.query(DbSeerRepoArchive)
                .filter(
                    DbSeerRepoArchive.organization_id == 1,
                    DbSeerRepoArchive.blob_path
                    == "repos/1/github/test-owner/test-repo_1234567890.tar.gz",
                )
                .count()
            )
            assert remaining_archives == 0


def test_delete_archive_no_organization_id(repo_manager):
    """Test that delete_archive raises RepoInitializationError when organization_id is None."""
    repo_manager.organization_id = None

    with pytest.raises(RepoInitializationError, match="Organization ID is not set"):
        repo_manager.delete_archive()


def test_delete_archive_gcs_blob_not_exists(repo_manager):
    """Test deletion when GCS blob doesn't exist but database record does."""
    from seer.automation.models import RepoDefinition
    from seer.db import DbSeerRepoArchive, Session

    # Create a test database record
    repo_definition = RepoDefinition(
        provider="github",
        owner="test-owner",
        name="test-repo",
        external_id="1234567890",
    )

    with Session() as session:
        repo_archive = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="test-bucket",
            blob_path="repos/1/github/test-owner/test-repo_1234567890.tar.gz",
            commit_sha="abcd123",
            repo_definition=repo_definition.model_dump(),
        )
        session.add(repo_archive)
        session.commit()

    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
    ):
        # Mock GCS blob that doesn't exist
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Execute delete_archive - should handle gracefully
        repo_manager.delete_archive()

        # Verify GCS operations were called
        mock_blob.exists.assert_called_once()
        mock_blob.delete.assert_not_called()  # Should not try to delete non-existent blob

        # Verify database record is still deleted
        with Session() as session:
            remaining_archives = (
                session.query(DbSeerRepoArchive)
                .filter(
                    DbSeerRepoArchive.organization_id == 1,
                    DbSeerRepoArchive.blob_path
                    == "repos/1/github/test-owner/test-repo_1234567890.tar.gz",
                )
                .count()
            )
            assert remaining_archives == 0


def test_delete_archive_db_record_not_exists(repo_manager):
    """Test deletion when database record doesn't exist but GCS blob does."""
    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
    ):
        # Mock GCS blob that exists
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Execute delete_archive - should handle gracefully
        repo_manager.delete_archive()

        # Verify GCS blob was deleted
        mock_blob.exists.assert_called_once()
        mock_blob.delete.assert_called_once()

        # No database record to verify deletion since none existed


def test_delete_archive_neither_exists(repo_manager):
    """Test deletion when neither GCS blob nor database record exist."""
    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
    ):
        # Mock GCS blob that doesn't exist
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Execute delete_archive - should handle gracefully
        repo_manager.delete_archive()

        # Verify GCS operations were called
        mock_blob.exists.assert_called_once()
        mock_blob.delete.assert_not_called()


def test_delete_archive_gcs_deletion_fails(repo_manager):
    """Test that GCS deletion failures are properly raised."""
    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
    ):
        # Mock GCS blob that raises exception on delete
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.delete.side_effect = Exception("GCS deletion failed")
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Execute delete_archive - should raise the exception
        with pytest.raises(Exception, match="GCS deletion failed"):
            repo_manager.delete_archive()


def test_delete_archive_db_deletion_fails(repo_manager):
    """Test that database deletion failures are properly raised."""
    from seer.automation.models import RepoDefinition
    from seer.db import DbSeerRepoArchive, Session

    # Create a test database record
    repo_definition = RepoDefinition(
        provider="github",
        owner="test-owner",
        name="test-repo",
        external_id="1234567890",
    )

    with Session() as session:
        repo_archive = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="test-bucket",
            blob_path="repos/1/github/test-owner/test-repo_1234567890.tar.gz",
            commit_sha="abcd123",
            repo_definition=repo_definition.model_dump(),
        )
        session.add(repo_archive)
        session.commit()

    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
        patch("seer.automation.codebase.repo_manager.Session") as mock_session_cls,
    ):
        # Mock successful GCS deletion
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Mock database session that raises exception on commit
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.commit.side_effect = Exception("Database deletion failed")

        # Execute delete_archive - should raise the exception
        with pytest.raises(Exception, match="Database deletion failed"):
            repo_manager.delete_archive()

        # Verify GCS blob was deleted before the DB failure
        mock_blob.delete.assert_called_once()


def test_delete_archive_realistic_scenario(repo_manager):
    """Test delete_archive with realistic repository data."""
    from seer.automation.models import RepoDefinition
    from seer.db import DbSeerRepoArchive, Session

    # Create realistic test data
    repo_definition = RepoDefinition(
        provider="github",
        owner="getsentry",
        name="sentry",
        external_id="4088350",
    )

    # Update repo_manager's repo_client to match the realistic data
    repo_manager.repo_client.provider = "github"
    repo_manager.repo_client.repo_owner = "getsentry"
    repo_manager.repo_client.repo_name = "sentry"
    repo_manager.repo_client.repo_external_id = "4088350"

    with Session() as session:
        repo_archive = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="sentry-code-analysis-prod",
            blob_path="repos/1/github/getsentry/sentry_4088350.tar.gz",
            commit_sha="a1b2c3d4e5f6789012345678901234567890abcd",
            repo_definition=repo_definition.model_dump(),
        )
        session.add(repo_archive)
        session.commit()

    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="sentry-code-analysis-prod"),
    ):
        # Mock realistic GCS operations
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Execute delete_archive
        repo_manager.delete_archive()

        # Verify operations were performed with realistic data
        mock_bucket.blob.assert_called_once_with("repos/1/github/getsentry/sentry_4088350.tar.gz")
        mock_blob.delete.assert_called_once()

        # Verify database cleanup
        with Session() as session:
            remaining_archives = (
                session.query(DbSeerRepoArchive)
                .filter(
                    DbSeerRepoArchive.organization_id == 1,
                    DbSeerRepoArchive.blob_path == "repos/1/github/getsentry/sentry_4088350.tar.gz",
                )
                .count()
            )
            assert remaining_archives == 0


def test_delete_archive_multiple_archives_same_org(repo_manager):
    """Test that delete_archive only deletes the specific archive, not others from same org."""
    from seer.automation.models import RepoDefinition
    from seer.db import DbSeerRepoArchive, Session

    # Create multiple archives for same organization
    repo_definition_1 = RepoDefinition(
        provider="github",
        owner="test-owner",
        name="test-repo-1",
        external_id="1111111111",
    )

    repo_definition_2 = RepoDefinition(
        provider="github",
        owner="test-owner",
        name="test-repo-2",
        external_id="2222222222",
    )

    with Session() as session:
        # Archive that should be deleted
        repo_archive_1 = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="test-bucket",
            blob_path="repos/1/github/test-owner/test-repo-1_1111111111.tar.gz",
            commit_sha="abcd123",
            repo_definition=repo_definition_1.model_dump(),
        )

        # Archive that should NOT be deleted
        repo_archive_2 = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="test-bucket",
            blob_path="repos/1/github/test-owner/test-repo-2_2222222222.tar.gz",
            commit_sha="efgh456",
            repo_definition=repo_definition_2.model_dump(),
        )

        session.add(repo_archive_1)
        session.add(repo_archive_2)
        session.commit()

    # Set repo_manager to point to the first archive
    repo_manager.repo_client.repo_name = "test-repo-1"
    repo_manager.repo_client.repo_external_id = "1111111111"

    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
    ):
        # Mock GCS blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Execute delete_archive
        repo_manager.delete_archive()

        # Verify only the first archive was deleted
        with Session() as session:
            remaining_archives = (
                session.query(DbSeerRepoArchive)
                .filter(DbSeerRepoArchive.organization_id == 1)
                .all()
            )

            # Should have exactly 1 remaining archive (the second one)
            assert len(remaining_archives) == 1
            assert (
                remaining_archives[0].blob_path
                == "repos/1/github/test-owner/test-repo-2_2222222222.tar.gz"
            )


def test_delete_archive_concurrent_deletion_safety(repo_manager):
    """Test that delete_archive handles concurrent deletion attempts safely."""
    from seer.automation.models import RepoDefinition
    from seer.db import DbSeerRepoArchive, Session

    # Create a test database record
    repo_definition = RepoDefinition(
        provider="github",
        owner="test-owner",
        name="test-repo",
        external_id="1234567890",
    )

    with Session() as session:
        repo_archive = DbSeerRepoArchive(
            organization_id=1,
            bucket_name="test-bucket",
            blob_path="repos/1/github/test-owner/test-repo_1234567890.tar.gz",
            commit_sha="abcd123",
            repo_definition=repo_definition.model_dump(),
        )
        session.add(repo_archive)
        session.commit()

    with (
        patch("seer.automation.codebase.repo_manager.storage.Client") as mock_storage_client,
        patch.object(repo_manager, "get_bucket_name", return_value="test-bucket"),
    ):
        # Mock GCS blob - simulate concurrent deletion by having exists() return False
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False  # Already deleted by another process
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Delete the database record manually to simulate concurrent deletion
        with Session() as session:
            session.query(DbSeerRepoArchive).filter(
                DbSeerRepoArchive.organization_id == 1,
                DbSeerRepoArchive.blob_path
                == "repos/1/github/test-owner/test-repo_1234567890.tar.gz",
            ).delete()
            session.commit()

        # Execute delete_archive - should handle gracefully when resources are already gone
        repo_manager.delete_archive()

        # Verify the calls were made even though resources were already deleted
        mock_blob.exists.assert_called_once()
        mock_blob.delete.assert_not_called()
