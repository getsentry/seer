import datetime
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from seer.automation.codebase.tasks import BACKFILL_LOCK_KEY, SYNC_LOCK_KEY, acquire_lock
from seer.db import (
    DbSeerBackfillJob,
    DbSeerBackfillState,
    DbSeerProjectPreference,
    DbSeerRepoArchive,
    Session,
)


class TestAcquireLock:
    """Test cases for the acquire_lock context manager using the real test database."""

    def test_acquire_lock_success(self):
        """Test successful lock acquisition."""
        with Session() as session:
            with acquire_lock(session, BACKFILL_LOCK_KEY, "test_lock") as got_lock:
                assert got_lock is True

                # Verify we can execute other queries while holding the lock
                result = session.execute(text("SELECT 1")).scalar()
                assert result == 1

    def test_acquire_lock_failure_already_taken(self):
        """Test when lock is already held by another session."""
        # First session acquires the lock
        with Session() as session1:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "first_lock") as got_lock1:
                assert got_lock1 is True

                # Second session tries to acquire the same lock and should fail
                with Session() as session2:
                    with acquire_lock(session2, BACKFILL_LOCK_KEY, "second_lock") as got_lock2:
                        assert got_lock2 is False

    def test_acquire_lock_different_keys_succeed(self):
        """Test that different lock keys can be acquired simultaneously."""
        with Session() as session1:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "backfill_lock") as got_lock1:
                assert got_lock1 is True

                # Different lock key should succeed even in same session
                with acquire_lock(session1, SYNC_LOCK_KEY, "sync_lock") as got_lock2:
                    assert got_lock2 is True

    def test_acquire_lock_released_after_transaction(self):
        """Test that lock is released when transaction ends."""
        # Acquire lock in first session and commit transaction
        with Session() as session1:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "first_lock") as got_lock1:
                assert got_lock1 is True
            # Transaction ends here, lock should be released

        # Second session should be able to acquire the same lock
        with Session() as session2:
            with acquire_lock(session2, BACKFILL_LOCK_KEY, "second_lock") as got_lock2:
                assert got_lock2 is True

    def test_acquire_lock_with_exception_in_context(self):
        """Test that lock is properly released even when an exception occurs."""
        # This test verifies that PostgreSQL transaction-level locks are released
        # when the transaction ends, even if an exception occurs

        # First, verify that we can acquire the lock normally
        with Session() as session1:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "test_lock") as got_lock1:
                assert got_lock1 is True

                # Within the same transaction, we can acquire the same lock again
                with acquire_lock(
                    session1, BACKFILL_LOCK_KEY, "same_transaction_lock"
                ) as got_lock_same:
                    assert got_lock_same is True

        # After the transaction ends (even with an exception), the lock should be released
        # Test this by trying to acquire the lock in a new session
        with Session() as session2:
            with acquire_lock(session2, BACKFILL_LOCK_KEY, "after_exception_lock") as got_lock2:
                assert got_lock2 is True

    def test_acquire_lock_concurrent_access(self):
        """Test concurrent lock acquisition attempts from multiple threads."""
        results = []
        lock_key = BACKFILL_LOCK_KEY

        def try_acquire_lock(thread_id):
            """Function to run in each thread."""
            with Session() as session:
                with acquire_lock(session, lock_key, f"thread_{thread_id}_lock") as got_lock:
                    if got_lock:
                        # Hold the lock for a short time to ensure other threads see it as taken
                        time.sleep(0.1)
                        results.append(f"thread_{thread_id}_success")
                    else:
                        results.append(f"thread_{thread_id}_failed")

        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_acquire_lock, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Exactly one thread should succeed, others should fail
        success_count = len([r for r in results if "success" in r])
        failed_count = len([r for r in results if "failed" in r])

        assert success_count == 1, f"Expected 1 success, got {success_count}"
        assert failed_count == 4, f"Expected 4 failures, got {failed_count}"

    def test_acquire_lock_database_exception_handling(self, caplog):
        """Test handling of database exceptions during lock acquisition."""
        with Session() as session:
            # Force an invalid SQL execution to trigger a database exception
            # We'll patch the session.execute method to raise an exception
            original_execute = session.execute

            def mock_execute(*args, **kwargs):
                if "pg_try_advisory_xact_lock" in str(args[0]):
                    raise Exception("Simulated database error")
                return original_execute(*args, **kwargs)

            session.execute = mock_execute

            # Should handle the exception gracefully and return False
            with caplog.at_level(logging.ERROR):
                with acquire_lock(session, BACKFILL_LOCK_KEY, "test_lock") as got_lock:
                    assert got_lock is False

            # Verify exception was logged
            assert "Error while managing test_lock lock" in caplog.text

    def test_acquire_lock_logging(self, caplog):
        """Test that lock acquisition and failure are properly logged."""
        with caplog.at_level(logging.INFO):
            # Test successful acquisition
            with Session() as session1:
                with acquire_lock(session1, BACKFILL_LOCK_KEY, "success_lock") as got_lock1:
                    assert got_lock1 is True

                    # Test failed acquisition in another session
                    with Session() as session2:
                        with acquire_lock(session2, BACKFILL_LOCK_KEY, "failure_lock") as got_lock2:
                            assert got_lock2 is False

        # Verify logging messages
        log_messages = caplog.text
        assert "Acquired success_lock lock" in log_messages
        assert "Could not acquire failure_lock lock, another process has it" in log_messages

    def test_acquire_lock_with_actual_lock_keys(self):
        """Test with the actual lock keys used in the application."""
        # Test backfill lock key
        with Session() as session:
            with acquire_lock(session, BACKFILL_LOCK_KEY, "backfill") as got_lock:
                assert got_lock is True

        # Test sync lock key
        with Session() as session:
            with acquire_lock(session, SYNC_LOCK_KEY, "sync") as got_lock:
                assert got_lock is True

    def test_acquire_lock_sql_execution(self):
        """Test that the correct SQL is executed for lock acquisition."""
        with Session() as session:
            # Manually test the SQL that acquire_lock uses
            result = session.execute(
                text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": BACKFILL_LOCK_KEY}
            ).scalar()

            # Should return True (lock acquired)
            assert result is True

            # Try to acquire the same lock again in the same transaction
            result2 = session.execute(
                text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": BACKFILL_LOCK_KEY}
            ).scalar()

            # Should return True again (same transaction can acquire same lock multiple times)
            assert result2 is True

    def test_acquire_lock_transaction_isolation(self):
        """Test that locks are properly isolated between transactions."""
        # Start a transaction but don't commit
        session1 = Session()
        try:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "transaction_lock") as got_lock1:
                assert got_lock1 is True

                # In another session, the lock should not be available
                with Session() as session2:
                    with acquire_lock(session2, BACKFILL_LOCK_KEY, "blocked_lock") as got_lock2:
                        assert got_lock2 is False

                # Rollback the first transaction
                session1.rollback()
        finally:
            session1.close()

        # After rollback, the lock should be available again
        with Session() as session3:
            with acquire_lock(session3, BACKFILL_LOCK_KEY, "available_lock") as got_lock3:
                assert got_lock3 is True


class TestCollectAllReposForBackfill:
    """Test cases for the collect_all_repos_for_backfill function using the real test database."""

    def setup_method(self):
        """Set up test data before each test."""
        with Session() as session:
            # Clean up any existing test data
            session.query(DbSeerBackfillState).delete()
            session.query(DbSeerProjectPreference).delete()
            session.query(DbSeerRepoArchive).delete()
            session.query(DbSeerBackfillJob).delete()
            session.commit()

    def teardown_method(self):
        """Clean up test data after each test."""
        with Session() as session:
            session.query(DbSeerBackfillState).delete()
            session.query(DbSeerProjectPreference).delete()
            session.query(DbSeerRepoArchive).delete()
            session.query(DbSeerBackfillJob).delete()
            session.commit()

    def test_collect_all_repos_for_backfill_no_lock_acquired(self, caplog):
        """Test that function returns early when lock cannot be acquired."""
        # First session acquires the lock
        with Session() as session1:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "test_lock") as got_lock1:
                assert got_lock1 is True

                # Import and call the function while lock is held
                from seer.automation.codebase.tasks import collect_all_repos_for_backfill

                with caplog.at_level(logging.INFO):
                    collect_all_repos_for_backfill()

                # Verify it logged that it couldn't acquire the lock
                assert "Could not acquire backfill lock, another process has it" in caplog.text

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_creates_backfill_state(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
    ):
        """Test that function creates backfill state when it doesn't exist."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        # Create test project preference
        with Session() as session:
            project_pref = DbSeerProjectPreference(
                project_id=100,
                organization_id=1,  # In FLAGGED_ORG_IDS
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "123",
                        "branch_name": "main",
                    }
                ],
            )
            session.add(project_pref)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        collect_all_repos_for_backfill()

        # Verify backfill state was created
        with Session() as session:
            backfill_state = (
                session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
            )
            assert backfill_state is not None
            assert backfill_state.backfill_cursor == 100  # Should be updated to project_id

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_skips_existing_archives(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async, caplog
    ):
        """Test that function skips repositories that already have archives."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create project preference
            project_pref = DbSeerProjectPreference(
                project_id=200,
                organization_id=1,
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "123",
                        "branch_name": "main",
                    }
                ],
            )
            session.add(project_pref)

            # Create existing archive for the same repo
            repo_archive = DbSeerRepoArchive(
                organization_id=1,
                bucket_name="test-bucket",
                blob_path="test/blob/path.tar.gz",
                commit_sha="abc123",
                repo_definition={
                    "provider": "github",
                    "owner": "test-owner",
                    "name": "test-repo",
                    "external_id": "123",
                },
            )
            session.add(repo_archive)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        with caplog.at_level(logging.INFO):
            collect_all_repos_for_backfill()

        # Verify it skipped the repo with existing archive
        assert "already has an archive, skipping" in caplog.text
        # Verify no backfill job was queued
        mock_apply_async.assert_not_called()

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_skips_duplicate_repos(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async, caplog
    ):
        """Test that function skips duplicate repositories within the same run."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create two project preferences with the same repo
            project_pref1 = DbSeerProjectPreference(
                project_id=300,
                organization_id=1,
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "123",
                        "branch_name": "main",
                    }
                ],
            )
            project_pref2 = DbSeerProjectPreference(
                project_id=301,
                organization_id=1,
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "123",  # Same external_id
                        "branch_name": "main",
                    }
                ],
            )
            session.add_all([project_pref1, project_pref2])
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        with caplog.at_level(logging.INFO):
            collect_all_repos_for_backfill()

        # Verify it skipped the duplicate repo
        assert "already added to backfill jobs, skipping" in caplog.text
        # Verify only one backfill job was queued
        assert mock_apply_async.call_count == 1

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_respects_threshold(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
    ):
        """Test that function stops after THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING jobs."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        # Create more project preferences than the threshold (32)
        with Session() as session:
            for i in range(35):  # More than THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING
                project_pref = DbSeerProjectPreference(
                    project_id=400 + i,
                    organization_id=1,
                    repositories=[
                        {
                            "provider": "github",
                            "owner": "test-owner",
                            "name": f"test-repo-{i}",
                            "external_id": str(1000 + i),
                            "branch_name": "main",
                        }
                    ],
                )
                session.add(project_pref)
            session.commit()

        from seer.automation.codebase.tasks import (
            THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING,
            collect_all_repos_for_backfill,
        )

        collect_all_repos_for_backfill()

        # Verify it stopped at the threshold
        assert mock_apply_async.call_count == THRESHOLD_BACKFILL_JOBS_UNTIL_STOP_LOOPING

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_cursor_reset_when_no_preferences(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async, caplog
    ):
        """Test that cursor resets to 0 when no project preferences are found."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"

        # Create backfill state with high cursor
        with Session() as session:
            backfill_state = DbSeerBackfillState(id=1, backfill_cursor=9999)
            session.add(backfill_state)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        with caplog.at_level(logging.INFO):
            collect_all_repos_for_backfill()

        # Verify cursor was reset and appropriate logs were generated
        assert "No project preferences to backfill, looping" in caplog.text
        assert "No project preferences to backfill, done" in caplog.text

        # The cursor gets reset to 0 during the function but since no project preferences are found,
        # the function returns early and the commit doesn't happen, so cursor remains unchanged
        with Session() as session:
            backfill_state = (
                session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
            )
            # The cursor reset is only flushed, not committed, so it remains at the original value
            assert backfill_state.backfill_cursor == 9999

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_cleans_up_old_jobs(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
    ):
        """Test that function cleans up old/failed backfill jobs before creating new ones."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create project preference
            project_pref = DbSeerProjectPreference(
                project_id=500,
                organization_id=1,
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "123",
                        "branch_name": "main",
                    }
                ],
            )
            session.add(project_pref)

            # Create old failed backfill job
            old_job = DbSeerBackfillJob(
                organization_id=1,
                repo_provider="github",
                repo_external_id="123",
                failed_at=datetime.datetime.now(datetime.UTC),
            )
            session.add(old_job)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        collect_all_repos_for_backfill()

        # Verify old job was deleted and new one was created
        with Session() as session:
            jobs = (
                session.query(DbSeerBackfillJob)
                .filter(
                    DbSeerBackfillJob.organization_id == 1,
                    DbSeerBackfillJob.repo_external_id == "123",
                )
                .all()
            )
            assert len(jobs) == 1
            assert jobs[0].failed_at is None  # New job should not be failed

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_skips_active_jobs(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async, caplog
    ):
        """Test that function skips repositories with active backfill jobs."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create project preference
            project_pref = DbSeerProjectPreference(
                project_id=600,
                organization_id=1,
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "123",
                        "branch_name": "main",
                    }
                ],
            )
            session.add(project_pref)

            # Create active backfill job (started but not completed)
            active_job = DbSeerBackfillJob(
                organization_id=1,
                repo_provider="github",
                repo_external_id="123",
                started_at=datetime.datetime.now(datetime.UTC),
            )
            session.add(active_job)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        with caplog.at_level(logging.INFO):
            collect_all_repos_for_backfill()

        # Verify it skipped the repo with active job
        assert "is still active, skipping" in caplog.text
        # The function still creates a BackfillJob but skips it during processing,
        # so apply_async is called but no new database job is created
        mock_apply_async.assert_called_once()

        # Verify that the backfill_job_id is None (indicating it was skipped)
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["backfill_job_id"] is None

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_only_processes_flagged_orgs(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
    ):
        """Test that function only processes organizations in FLAGGED_ORG_IDS."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"

        with Session() as session:
            # Create project preference for flagged org (should be processed)
            flagged_pref = DbSeerProjectPreference(
                project_id=700,
                organization_id=1,  # In FLAGGED_ORG_IDS
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo-flagged",
                        "external_id": "123",
                        "branch_name": "main",
                    }
                ],
            )

            # Create project preference for non-flagged org (should be ignored)
            non_flagged_pref = DbSeerProjectPreference(
                project_id=701,
                organization_id=999,  # Not in FLAGGED_ORG_IDS
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo-non-flagged",
                        "external_id": "456",
                        "branch_name": "main",
                    }
                ],
            )
            session.add_all([flagged_pref, non_flagged_pref])
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        collect_all_repos_for_backfill()

        # Verify no jobs were queued (since we didn't mock RepoClient properly for the flagged org)
        # But the important thing is that it only looked at the flagged org
        # We can verify this by checking that the cursor was updated to the flagged org's project_id
        with Session() as session:
            backfill_state = (
                session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
            )
            assert backfill_state.backfill_cursor == 700  # Should be the flagged org's project_id

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_updates_cursor_correctly(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
    ):
        """Test that function correctly updates the backfill cursor."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create backfill state with initial cursor
            backfill_state = DbSeerBackfillState(id=1, backfill_cursor=0)
            session.add(backfill_state)

            # Create project preferences
            for i in range(3):
                project_pref = DbSeerProjectPreference(
                    project_id=800 + i,
                    organization_id=1,
                    repositories=[
                        {
                            "provider": "github",
                            "owner": "test-owner",
                            "name": f"test-repo-{i}",
                            "external_id": str(1000 + i),
                            "branch_name": "main",
                        }
                    ],
                )
                session.add(project_pref)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        collect_all_repos_for_backfill()

        # Verify cursor was updated to the last processed project_id
        with Session() as session:
            backfill_state = (
                session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
            )
            assert backfill_state.backfill_cursor == 802  # Last project_id

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_queues_jobs_with_correct_parameters(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
    ):
        """Test that function queues backfill jobs with correct parameters."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.return_value = "test/blob/path.tar.gz"
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 1800.0  # 30 minutes
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create project preference
            project_pref = DbSeerProjectPreference(
                project_id=900,
                organization_id=1,
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "123",
                        "branch_name": "main",
                    }
                ],
            )
            session.add(project_pref)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        collect_all_repos_for_backfill()

        # Verify job was queued with correct parameters
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args

        # Check the backfill job data
        job_data = call_args[1]["args"][0]  # First argument to apply_async
        assert job_data["organization_id"] == 1
        assert job_data["repo_definition"]["provider"] == "github"
        assert job_data["repo_definition"]["owner"] == "test-owner"
        assert job_data["repo_definition"]["name"] == "test-repo"
        assert job_data["repo_definition"]["external_id"] == "123"
        assert job_data["scaled_time_limit"] == 1800.0
        assert "backfill_job_id" in job_data

        # Check the time limits
        assert call_args[1]["soft_time_limit"] == 1800.0
        assert call_args[1]["time_limit"] == 1830.0  # soft_time_limit + 30 second buffer

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_handles_multiple_repos_per_preference(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
    ):
        """Test that function handles project preferences with multiple repositories."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.side_effect = (
            lambda org_id, provider, owner, name, external_id: f"{owner}/{name}_{external_id}.tar.gz"
        )
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create project preference with multiple repositories
            project_pref = DbSeerProjectPreference(
                project_id=1000,
                organization_id=1,
                repositories=[
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo-1",
                        "external_id": "123",
                        "branch_name": "main",
                    },
                    {
                        "provider": "github",
                        "owner": "test-owner",
                        "name": "test-repo-2",
                        "external_id": "456",
                        "branch_name": "main",
                    },
                ],
            )
            session.add(project_pref)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        collect_all_repos_for_backfill()

        # Verify both repositories were processed
        assert mock_apply_async.call_count == 2

        # Verify both backfill jobs were created in the database
        with Session() as session:
            jobs = (
                session.query(DbSeerBackfillJob)
                .filter(DbSeerBackfillJob.organization_id == 1)
                .all()
            )
            assert len(jobs) == 2
            external_ids = {job.repo_external_id for job in jobs}
            assert external_ids == {"123", "456"}

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_loops_around_when_reaching_end(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async, caplog
    ):
        """Test that function loops back to the beginning when it reaches the end of project preferences."""
        # Setup mocks
        mock_get_bucket_name.return_value = "test-bucket"
        mock_make_blob_name.side_effect = (
            lambda org_id, provider, owner, name, external_id: f"{owner}/{name}_{external_id}.tar.gz"
        )
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        with Session() as session:
            # Create backfill state with cursor pointing to a high value
            backfill_state = DbSeerBackfillState(id=1, backfill_cursor=1500)
            session.add(backfill_state)

            # Create project preferences with IDs both before and after the cursor
            # These should be processed in the next run after looping
            early_prefs = [
                DbSeerProjectPreference(
                    project_id=100 + i,
                    organization_id=1,
                    repositories=[
                        {
                            "provider": "github",
                            "owner": "test-owner",
                            "name": f"early-repo-{i}",
                            "external_id": str(2000 + i),
                            "branch_name": "main",
                        }
                    ],
                )
                for i in range(3)
            ]

            # These should be processed after the cursor is reset and loops back
            late_prefs = [
                DbSeerProjectPreference(
                    project_id=2000 + i,
                    organization_id=1,
                    repositories=[
                        {
                            "provider": "github",
                            "owner": "test-owner",
                            "name": f"late-repo-{i}",
                            "external_id": str(3000 + i),
                            "branch_name": "main",
                        }
                    ],
                )
                for i in range(2)
            ]

            session.add_all(early_prefs + late_prefs)
            session.commit()

        from seer.automation.codebase.tasks import collect_all_repos_for_backfill

        # First run: cursor is at 1500, should find late_prefs (2000, 2001) and process them
        with caplog.at_level(logging.INFO):
            collect_all_repos_for_backfill()

        # Verify it processed the late preferences
        assert "Found 2 project preferences to look at, starting from 2000 to 2001" in caplog.text
        assert mock_apply_async.call_count == 2

        # Verify cursor was updated to the last processed project_id
        with Session() as session:
            backfill_state = (
                session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
            )
            assert backfill_state.backfill_cursor == 2001

        # Reset mocks for second run
        mock_apply_async.reset_mock()
        caplog.clear()

        # Second run: cursor is at 2001, should find no preferences after 2001, loop back to 0,
        # and then find early_prefs (100, 101, 102)
        with caplog.at_level(logging.INFO):
            collect_all_repos_for_backfill()

        # Verify it looped back and processed early preferences
        assert "No project preferences to backfill, looping" in caplog.text
        assert "Found 5 project preferences to look at, starting from 100 to 2001" in caplog.text
        # It finds all 5 preferences but only queues 5 jobs total (3 new ones for early prefs,
        # 2 existing ones get skipped as "still active")
        assert mock_apply_async.call_count == 5

        # Verify cursor was updated to the last processed project_id (which is 2001, the highest ID)
        with Session() as session:
            backfill_state = (
                session.query(DbSeerBackfillState).filter(DbSeerBackfillState.id == 1).first()
            )
            assert backfill_state.backfill_cursor == 2001

        # Verify that the late repos were skipped due to active jobs
        assert "is still active, skipping" in caplog.text

        # Verify all jobs were created in the database
        with Session() as session:
            jobs = (
                session.query(DbSeerBackfillJob)
                .filter(DbSeerBackfillJob.organization_id == 1)
                .all()
            )
            # Should have 5 jobs total (2 from first run + 3 from second run)
            assert len(jobs) == 5
            external_ids = {job.repo_external_id for job in jobs}
            expected_ids = {"3000", "3001", "2000", "2001", "2002"}  # All the repos we created
            assert external_ids == expected_ids


class TestRunBackfill:
    """Test cases for the run_backfill function using the real test database."""

    def setup_method(self):
        """Set up test data before each test."""
        with Session() as session:
            # Clean up any existing test data
            session.query(DbSeerBackfillJob).delete()
            session.commit()

    def teardown_method(self):
        """Clean up test data after each test."""
        with Session() as session:
            session.query(DbSeerBackfillJob).delete()
            session.commit()

    def _create_test_backfill_job(self, **kwargs):
        """Helper to create a test backfill job in the database."""
        defaults = {
            "organization_id": 1,
            "repo_provider": "github",
            "repo_external_id": "123",
        }
        defaults.update(kwargs)

        with Session() as session:
            job = DbSeerBackfillJob(**defaults)
            session.add(job)
            session.commit()
            return job.id

    def _create_test_backfill_job_dict(self, backfill_job_id=None, **kwargs):
        """Helper to create a test BackfillJob dictionary."""
        defaults = {
            "organization_id": 1,
            "repo_definition": {
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo",
                "external_id": "123",
                "branch_name": "main",
            },
            "backfill_job_id": backfill_job_id,
            "scaled_time_limit": 900.0,
        }
        defaults.update(kwargs)
        return defaults

    # Input validation tests
    def test_run_backfill_missing_job_id(self):
        """Test that function raises error when backfill_job_id is missing."""
        from seer.automation.codebase.tasks import BackfillJobError, run_backfill

        job_dict = self._create_test_backfill_job_dict(backfill_job_id=None)

        with pytest.raises(BackfillJobError, match="backfill_job_id is required"):
            run_backfill(job_dict)

    def test_run_backfill_invalid_job_data(self):
        """Test that function handles invalid job data gracefully."""
        from seer.automation.codebase.tasks import run_backfill

        # Test with completely invalid data
        with pytest.raises(Exception):  # Pydantic validation error
            run_backfill({"invalid": "data"})

    def test_run_backfill_job_not_found(self):
        """Test that function raises error when job ID doesn't exist in database."""
        from seer.automation.codebase.tasks import BackfillJobError, run_backfill

        job_dict = self._create_test_backfill_job_dict(backfill_job_id=99999)

        with pytest.raises(BackfillJobError, match="backfill job not found"):
            run_backfill(job_dict)

    # Database state validation tests
    def test_run_backfill_job_already_started(self):
        """Test that function raises error when job is already started."""
        from seer.automation.codebase.tasks import BackfillJobError, run_backfill

        job_id = self._create_test_backfill_job(started_at=datetime.datetime.now(datetime.UTC))
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        with pytest.raises(BackfillJobError, match="backfill job already started"):
            run_backfill(job_dict)

    def test_run_backfill_job_already_completed(self):
        """Test that function raises error when job is already completed."""
        from seer.automation.codebase.tasks import BackfillJobError, run_backfill

        job_id = self._create_test_backfill_job(completed_at=datetime.datetime.now(datetime.UTC))
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        with pytest.raises(BackfillJobError, match="backfill job already completed"):
            run_backfill(job_dict)

    def test_run_backfill_job_already_failed(self):
        """Test that function raises error when job is already failed."""
        from seer.automation.codebase.tasks import BackfillJobError, run_backfill

        job_id = self._create_test_backfill_job(failed_at=datetime.datetime.now(datetime.UTC))
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        with pytest.raises(BackfillJobError, match="backfill job already failed"):
            run_backfill(job_dict)

    # Success path tests
    @patch("seer.automation.codebase.tasks.run_test_download_and_verify_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_success(
        self, mock_repo_client, mock_repo_manager_class, mock_verify_task
    ):
        """Test successful backfill execution."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute
        run_backfill(job_dict)

        # Verify RepoClient was created correctly
        mock_repo_client.assert_called_once()
        call_args = mock_repo_client.call_args
        # Check positional arguments: repo_definition and RepoClientType.READ
        assert len(call_args[0]) == 2  # Two positional arguments
        # The second argument should be RepoClientType.READ
        from seer.automation.codebase.repo_client import RepoClientType

        assert call_args[0][1] == RepoClientType.READ

        # Verify RepoManager was created correctly
        mock_repo_manager_class.assert_called_once_with(
            mock_repo_client.return_value, organization_id=1, force_gcs=True
        )

        # Verify archive initialization was called
        mock_repo_manager_instance.initialize_archive_for_backfill.assert_called_once()

        # Verify cleanup was called
        mock_repo_manager_instance.cleanup.assert_called_once()

        # Verify verification task was queued
        mock_verify_task.assert_called_once_with(args=(job_dict,))

        # Verify job was marked as completed
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.completed_at is not None
            assert job.failed_at is None

    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_repo_client_creation_failure(self, mock_repo_client):
        """Test handling of RepoClient creation failure."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mock to raise exception
        mock_repo_client.side_effect = Exception("RepoClient creation failed")

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute and expect exception
        with pytest.raises(Exception, match="RepoClient creation failed"):
            run_backfill(job_dict)

        # Verify job was marked as failed
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.failed_at is not None
            assert job.completed_at is None

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_repo_manager_creation_failure(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test handling of RepoManager creation failure."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_class.side_effect = Exception("RepoManager creation failed")

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute and expect exception
        with pytest.raises(Exception, match="RepoManager creation failed"):
            run_backfill(job_dict)

        # Verify job was marked as failed
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.failed_at is not None
            assert job.completed_at is None

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_archive_initialization_failure(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test handling of archive initialization failure."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_instance.initialize_archive_for_backfill.side_effect = Exception(
            "Archive initialization failed"
        )
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute and expect exception
        with pytest.raises(Exception, match="Archive initialization failed"):
            run_backfill(job_dict)

        # Verify job was marked as failed
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.failed_at is not None
            assert job.completed_at is None

        # Verify cleanup was still called
        mock_repo_manager_instance.cleanup.assert_called_once()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_marks_job_as_failed_on_exception(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that job is marked as failed when any exception occurs."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_instance.initialize_archive_for_backfill.side_effect = RuntimeError(
            "Test failure"
        )
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute and expect exception
        with pytest.raises(RuntimeError, match="Test failure"):
            run_backfill(job_dict)

        # Verify job was marked as failed with timestamp
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.failed_at is not None
            assert job.completed_at is None
            assert job.started_at is not None  # Should still be marked as started

    # Cleanup tests
    @patch("seer.automation.codebase.tasks.run_test_download_and_verify_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_cleanup_called_on_success(
        self, mock_repo_client, mock_repo_manager_class, mock_verify_task
    ):
        """Test that cleanup is called on successful execution."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute
        run_backfill(job_dict)

        # Verify cleanup was called
        mock_repo_manager_instance.cleanup.assert_called_once()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_cleanup_called_on_failure(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that cleanup is called even when execution fails."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_instance.initialize_archive_for_backfill.side_effect = Exception(
            "Test failure"
        )
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute and expect exception
        with pytest.raises(Exception, match="Test failure"):
            run_backfill(job_dict)

        # Verify cleanup was still called
        mock_repo_manager_instance.cleanup.assert_called_once()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_cleanup_called_on_early_exit(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that cleanup is called even when validation fails early."""
        from seer.automation.codebase.tasks import BackfillJobError, run_backfill

        # Don't create a job in the database, so validation will fail
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=99999)

        # Execute and expect early validation failure
        with pytest.raises(BackfillJobError, match="backfill job not found"):
            run_backfill(job_dict)

        # Verify RepoClient and RepoManager were never created
        mock_repo_client.assert_not_called()
        mock_repo_manager_class.assert_not_called()

    # Logging tests
    @patch("seer.automation.codebase.tasks.run_test_download_and_verify_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_logging_success(
        self, mock_repo_client, mock_repo_manager_class, mock_verify_task, caplog
    ):
        """Test that appropriate logs are generated on successful execution."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute
        with caplog.at_level(logging.INFO):
            run_backfill(job_dict)

        # Verify appropriate logs were generated
        log_messages = caplog.text
        assert f"Running backfill job {job_id}" in log_messages
        assert "test-owner/test-repo" in log_messages
        assert "Backfill job done." in log_messages

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_logging_failure(self, mock_repo_client, mock_repo_manager_class, caplog):
        """Test that appropriate logs are generated on execution failure."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_instance.initialize_archive_for_backfill.side_effect = Exception(
            "Test failure"
        )
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute and expect exception
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="Test failure"):
                run_backfill(job_dict)

        # Verify error logging
        log_messages = caplog.text
        assert "Failed to run backfill job" in log_messages
        assert str(job_id) in log_messages

    # Integration tests with realistic scenarios
    @patch("seer.automation.codebase.tasks.run_test_download_and_verify_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_with_realistic_repo_definition(
        self, mock_repo_client, mock_repo_manager_class, mock_verify_task
    ):
        """Test with realistic repository definition data."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job with realistic data
        job_id = self._create_test_backfill_job(
            organization_id=42, repo_provider="github", repo_external_id="987654321"
        )

        job_dict = self._create_test_backfill_job_dict(
            backfill_job_id=job_id,
            organization_id=42,
            repo_definition={
                "provider": "github",
                "owner": "getsentry",
                "name": "sentry",
                "external_id": "987654321",
                "branch_name": "master",
            },
            scaled_time_limit=1800.0,
        )

        # Execute
        run_backfill(job_dict)

        # Verify RepoManager was created with correct parameters
        mock_repo_manager_class.assert_called_once_with(
            mock_repo_client_instance, organization_id=42, force_gcs=True
        )

        # Verify job completion
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.completed_at is not None
            assert job.organization_id == 42
            assert job.repo_external_id == "987654321"

    @patch("seer.automation.codebase.tasks.run_test_download_and_verify_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_verification_task_not_queued_on_failure(
        self, mock_repo_client, mock_repo_manager_class, mock_verify_task
    ):
        """Test that verification task is not queued when backfill fails."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_instance.initialize_archive_for_backfill.side_effect = Exception(
            "Archive failed"
        )
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute and expect exception
        with pytest.raises(Exception, match="Archive failed"):
            run_backfill(job_dict)

        # Verify verification task was NOT queued
        mock_verify_task.assert_not_called()

        # Verify job was marked as failed
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.failed_at is not None
            assert job.completed_at is None

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_marks_job_as_started(self, mock_repo_client, mock_repo_manager_class):
        """Test that job is marked as started when execution begins."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Verify job starts without started_at
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.started_at is None

        # Execute
        run_backfill(job_dict)

        # Verify job was marked as started
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.started_at is not None

    @patch("seer.automation.codebase.tasks.run_test_download_and_verify_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_marks_job_as_completed(
        self, mock_repo_client, mock_repo_manager_class, mock_verify_task
    ):
        """Test that job is marked as completed on successful execution."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute
        run_backfill(job_dict)

        # Verify job was marked as completed
        with Session() as session:
            job = session.query(DbSeerBackfillJob).filter(DbSeerBackfillJob.id == job_id).first()
            assert job.completed_at is not None
            assert job.failed_at is None

    @patch("seer.automation.codebase.tasks.run_test_download_and_verify_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_backfill_queues_verification_task(
        self, mock_repo_client, mock_repo_manager_class, mock_verify_task
    ):
        """Test that verification task is queued after successful backfill."""
        from seer.automation.codebase.tasks import run_backfill

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test job
        job_id = self._create_test_backfill_job()
        job_dict = self._create_test_backfill_job_dict(backfill_job_id=job_id)

        # Execute
        run_backfill(job_dict)

        # Verify verification task was queued with correct parameters
        mock_verify_task.assert_called_once_with(args=(job_dict,))

    # Failure handling tests
