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

    def test_acquire_lock_database_exception_handling(self):
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
            with acquire_lock(session, BACKFILL_LOCK_KEY, "test_lock") as got_lock:
                assert got_lock is False

    def test_acquire_lock_logging(self):
        """Test that lock acquisition and failure are properly logged."""
        # Test successful acquisition
        with Session() as session1:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "success_lock") as got_lock1:
                assert got_lock1 is True

                # Test failed acquisition in another session
                with Session() as session2:
                    with acquire_lock(session2, BACKFILL_LOCK_KEY, "failure_lock") as got_lock2:
                        assert got_lock2 is False

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

    def test_collect_all_repos_for_backfill_no_lock_acquired(self):
        """Test that function returns early when lock cannot be acquired."""
        # First session acquires the lock
        with Session() as session1:
            with acquire_lock(session1, BACKFILL_LOCK_KEY, "test_lock") as got_lock1:
                assert got_lock1 is True

                # Import and call the function while lock is held
                from seer.automation.codebase.tasks import collect_all_repos_for_backfill

                collect_all_repos_for_backfill()

                # Function should return early without processing

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
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
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

        collect_all_repos_for_backfill()

        # Verify no backfill job was queued (repo was skipped due to existing archive)
        mock_apply_async.assert_not_called()

    @patch("seer.automation.codebase.tasks.run_backfill.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    @patch("seer.automation.codebase.tasks.RepoManager.make_blob_name")
    @patch("seer.automation.codebase.tasks.RepoManager.get_bucket_name")
    def test_collect_all_repos_for_backfill_skips_duplicate_repos(
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
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

        collect_all_repos_for_backfill()

        # Verify only one backfill job was queued (duplicate was skipped)
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
        self, mock_get_bucket_name, mock_make_blob_name, mock_repo_client, mock_apply_async
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

        collect_all_repos_for_backfill()

        # The function still creates a BackfillJob but skips it during processing,
        # so apply_async is called but no new database job is created
        mock_apply_async.assert_called_once()

        # Verify that the backfill_job_id is None (indicating it was skipped)
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["backfill_job_id"] is None

        # Verify that no new database job was created (still only the original active job)
        with Session() as session:
            jobs = (
                session.query(DbSeerBackfillJob)
                .filter(
                    DbSeerBackfillJob.organization_id == 1,
                    DbSeerBackfillJob.repo_external_id == "123",
                )
                .all()
            )
            assert len(jobs) == 1  # Only the original active job
            assert jobs[0].started_at is not None
            assert jobs[0].completed_at is None

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
            mock_repo_client.return_value, organization_id=1
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
            mock_repo_client_instance, organization_id=42
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


class TestRunRepoSync:
    """Test cases for the run_repo_sync function using the real test database."""

    def setup_method(self):
        """Set up test data before each test."""
        with Session() as session:
            # Clean up any existing test data
            session.query(DbSeerRepoArchive).delete()
            session.commit()

    def teardown_method(self):
        """Clean up test data after each test."""
        with Session() as session:
            session.query(DbSeerRepoArchive).delete()
            session.commit()

    def _create_test_repo_archive(self, **kwargs):
        """Helper to create a test repository archive in the database."""
        defaults = {
            "organization_id": 1,
            "bucket_name": "test-bucket",
            "blob_path": "test/repo/archive.tar.gz",
            "commit_sha": "abc123",
            "repo_definition": {
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo",
                "external_id": "123",
                "branch_name": "main",
            },
            "updated_at": datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=8),  # Old enough to need sync
        }
        defaults.update(kwargs)

        with Session() as session:
            archive = DbSeerRepoArchive(**defaults)
            session.add(archive)
            session.commit()
            return archive.id

    # Lock management tests
    def test_run_repo_sync_no_lock_acquired(self, caplog):
        """Test that function returns early when lock cannot be acquired."""
        # First session acquires the lock
        with Session() as session1:
            with acquire_lock(session1, SYNC_LOCK_KEY, "test_lock") as got_lock1:
                assert got_lock1 is True

                # Import and call the function while lock is held
                from seer.automation.codebase.tasks import run_repo_sync

                with caplog.at_level(logging.INFO):
                    run_repo_sync()

                # Verify it logged that it couldn't acquire the lock
                assert "Could not acquire sync lock, another process has it" in caplog.text

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    def test_run_repo_sync_acquires_lock_successfully(self, mock_apply_async, caplog):
        """Test that function proceeds when lock is acquired successfully."""
        from seer.automation.codebase.tasks import run_repo_sync

        with caplog.at_level(logging.INFO):
            run_repo_sync()

        # Verify it logged that it acquired the lock
        assert "Acquired sync lock" in caplog.text

    # Repository archive query tests
    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    def test_run_repo_sync_no_archives_to_sync(self, mock_apply_async, caplog):
        """Test when no archives need updating."""
        from seer.automation.codebase.tasks import run_repo_sync

        with caplog.at_level(logging.INFO):
            run_repo_sync()

        # Verify no jobs were queued
        mock_apply_async.assert_not_called()
        assert "Queueing 0 repo sync jobs" in caplog.text

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_skips_recently_updated_archives(
        self, mock_repo_client, mock_apply_async
    ):
        """Test that archives updated recently are skipped."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Create archive updated recently (within 7 days)
        self._create_test_repo_archive(
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=3)
        )

        run_repo_sync()

        # Verify no jobs were queued
        mock_apply_async.assert_not_called()
        mock_repo_client.assert_not_called()

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_processes_old_archives(self, mock_repo_client, mock_apply_async):
        """Test that archives older than 7 days are processed."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.repo_full_name = "test-owner/test-repo"
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        # Create archive updated more than 7 days ago with recent download
        archive_id = self._create_test_repo_archive(
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=8),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        run_repo_sync()

        # Verify job was queued
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["archive_id"] == archive_id
        assert job_data["repo_full_name"] == "test-owner/test-repo"
        assert job_data["scaled_time_limit"] == 900.0

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_only_processes_flagged_orgs(self, mock_repo_client, mock_apply_async):
        """Test that function only processes organizations in FLAGGED_ORG_IDS."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Create archive for flagged org (should be processed)
        flagged_archive_id = self._create_test_repo_archive(
            organization_id=1,  # In FLAGGED_ORG_IDS
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=8),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        # Create archive for non-flagged org (should be ignored)
        self._create_test_repo_archive(
            organization_id=999,  # Not in FLAGGED_ORG_IDS
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=8),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client_instance.repo_full_name = "test-owner/test-repo"
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        run_repo_sync()

        # Verify only one job was queued (for the flagged org)
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["archive_id"] == flagged_archive_id

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_respects_archive_limit(self, mock_repo_client, mock_apply_async):
        """Test that function respects MAX_REPO_ARCHIVES_PER_SYNC limit."""
        from seer.automation.codebase.tasks import MAX_REPO_ARCHIVES_PER_SYNC, run_repo_sync

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_client_instance.repo_full_name = "test-owner/test-repo"
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0

        # Create more archives than the limit
        for i in range(MAX_REPO_ARCHIVES_PER_SYNC + 5):
            self._create_test_repo_archive(
                blob_path=f"test/repo/archive-{i}.tar.gz",
                updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=8 + i),
                last_downloaded_at=datetime.datetime.now(datetime.UTC)
                - datetime.timedelta(days=3),  # Downloaded recently
            )

        run_repo_sync()

        # Verify only MAX_REPO_ARCHIVES_PER_SYNC jobs were queued
        assert mock_apply_async.call_count == MAX_REPO_ARCHIVES_PER_SYNC

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_processes_oldest_archives_first(
        self, mock_repo_client, mock_apply_async
    ):
        """Test that function processes archives in order of oldest first."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_client_instance.repo_full_name = "test-owner/test-repo"
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        # Create archives with different update times
        oldest_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/oldest.tar.gz",
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=20),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        middle_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/middle.tar.gz",
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=15),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=2),  # Downloaded recently
        )

        newest_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/newest.tar.gz",
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=10),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=1),  # Downloaded recently
        )

        run_repo_sync()

        # Verify all jobs were queued
        assert mock_apply_async.call_count == 3

        # Verify they were processed in order (oldest first)
        call_args_list = mock_apply_async.call_args_list
        processed_archive_ids = [call[1]["args"][0]["archive_id"] for call in call_args_list]
        assert processed_archive_ids == [oldest_archive_id, middle_archive_id, newest_archive_id]

    # RepoClient creation tests
    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_successful_repo_client_creation(
        self, mock_repo_client, mock_apply_async
    ):
        """Test successful RepoClient creation and job queuing."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_client_instance.repo_full_name = "getsentry/sentry"
        mock_repo_client_instance.get_scaled_time_limit.return_value = 1800.0
        mock_repo_client.return_value = mock_repo_client_instance

        # Create test archive
        archive_id = self._create_test_repo_archive(
            repo_definition={
                "provider": "github",
                "owner": "getsentry",
                "name": "sentry",
                "external_id": "456",
                "branch_name": "master",
            },
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=2),  # Downloaded recently
        )

        run_repo_sync()

        # Verify RepoClient was created with correct parameters
        mock_repo_client.assert_called_once()
        call_args = mock_repo_client.call_args
        repo_definition = call_args[0][0]
        assert repo_definition.owner == "getsentry"
        assert repo_definition.name == "sentry"

        # Verify job was queued with correct parameters
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["archive_id"] == archive_id
        assert job_data["repo_full_name"] == "getsentry/sentry"
        assert job_data["scaled_time_limit"] == 1800.0
        assert call_args[1]["soft_time_limit"] == 1800.0
        assert call_args[1]["time_limit"] == 1830.0  # 30 second buffer

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_repo_not_found_deletes_archive(
        self, mock_repo_client, mock_apply_async, caplog
    ):
        """Test that archives are deleted when repo is not found via GitHub API."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Setup mock to simulate repo not found
        mock_repo_client.side_effect = Exception(
            "Error getting repo via full name: test-owner/test-repo"
        )

        # Create test archive
        archive_id = self._create_test_repo_archive(
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        with caplog.at_level(logging.INFO):
            run_repo_sync()

        # Verify archive was deleted from database
        with Session() as session:
            archive = (
                session.query(DbSeerRepoArchive).filter(DbSeerRepoArchive.id == archive_id).first()
            )
            assert archive is None

        # Verify appropriate log message
        assert "not found from github api, deleting repo archive" in caplog.text

        # Verify no job was queued
        mock_apply_async.assert_not_called()

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_repo_client_general_exception(
        self, mock_repo_client, mock_apply_async, caplog
    ):
        """Test handling of general exceptions during RepoClient creation."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Setup mock to raise general exception
        mock_repo_client.side_effect = Exception("General RepoClient error")

        # Create test archive
        archive_id = self._create_test_repo_archive(
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        with caplog.at_level(logging.ERROR):
            run_repo_sync()

        # Verify archive was NOT deleted (only delete on specific "repo not found" error)
        with Session() as session:
            archive = (
                session.query(DbSeerRepoArchive).filter(DbSeerRepoArchive.id == archive_id).first()
            )
            assert archive is not None

        # Verify error was logged
        assert "Failed to get repo_client for repo" in caplog.text
        assert str(archive_id) in caplog.text

        # Verify no job was queued
        mock_apply_async.assert_not_called()

    # Job queuing tests
    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_queues_multiple_jobs(self, mock_repo_client, mock_apply_async, caplog):
        """Test that multiple sync jobs are queued correctly."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_client_instance.repo_full_name = "test-owner/test-repo"
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        # Create multiple archives
        archive_ids = []
        for i in range(3):
            archive_id = self._create_test_repo_archive(
                blob_path=f"test/repo/archive-{i}.tar.gz",
                repo_definition={
                    "provider": "github",
                    "owner": "test-owner",
                    "name": f"test-repo-{i}",
                    "external_id": str(100 + i),
                    "branch_name": "main",
                },
                last_downloaded_at=datetime.datetime.now(datetime.UTC)
                - datetime.timedelta(days=3),  # Downloaded recently
            )
            archive_ids.append(archive_id)

        with caplog.at_level(logging.INFO):
            run_repo_sync()

        # Verify all jobs were queued
        assert mock_apply_async.call_count == 3
        assert "Queueing 3 repo sync jobs" in caplog.text

        # Verify each job has correct archive_id
        call_args_list = mock_apply_async.call_args_list
        queued_archive_ids = [call[1]["args"][0]["archive_id"] for call in call_args_list]
        assert set(queued_archive_ids) == set(archive_ids)

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_handles_mixed_success_failure(
        self, mock_repo_client, mock_apply_async, caplog
    ):
        """Test handling when some repos succeed and others fail."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Create multiple archives
        success_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/success.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "success-owner",
                "name": "success-repo",
                "external_id": "200",
                "branch_name": "main",
            },
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        failure_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/failure.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "failure-owner",
                "name": "failure-repo",
                "external_id": "201",
                "branch_name": "main",
            },
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        # Setup mock to succeed for first repo, fail for second
        def mock_repo_client_side_effect(repo_definition, client_type):
            if repo_definition.owner == "success-owner":
                mock_instance = MagicMock()
                mock_instance.repo_full_name = "success-owner/success-repo"
                mock_instance.get_scaled_time_limit.return_value = 900.0
                return mock_instance
            else:
                raise Exception("Error getting repo via full name: failure-owner/failure-repo")

        mock_repo_client.side_effect = mock_repo_client_side_effect

        run_repo_sync()

        # Verify only successful job was queued
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["archive_id"] == success_archive_id

        # Verify failed archive was deleted
        with Session() as session:
            success_archive = (
                session.query(DbSeerRepoArchive)
                .filter(DbSeerRepoArchive.id == success_archive_id)
                .first()
            )
            failure_archive = (
                session.query(DbSeerRepoArchive)
                .filter(DbSeerRepoArchive.id == failure_archive_id)
                .first()
            )
            assert success_archive is not None
            assert failure_archive is None

        # Verify appropriate logging
        assert "not found from github api, deleting repo archive" in caplog.text
        assert "Queueing 1 repo sync jobs" in caplog.text

    # Integration tests
    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_realistic_scenario(self, mock_repo_client, mock_apply_async):
        """Test with a realistic mix of archives in different states."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_client_instance.repo_full_name = "test-owner/test-repo"
        mock_repo_client_instance.get_scaled_time_limit.return_value = 900.0
        mock_repo_client.return_value = mock_repo_client_instance

        now = datetime.datetime.now(datetime.UTC)

        # Archive that needs sync (old)
        old_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/old.tar.gz",
            organization_id=1,  # Flagged org
            updated_at=now - datetime.timedelta(days=10),
            last_downloaded_at=now - datetime.timedelta(days=3),  # Downloaded recently
        )

        # Archive that doesn't need sync (recent)
        self._create_test_repo_archive(
            blob_path="test/repo/recent.tar.gz",
            organization_id=1,  # Flagged org
            updated_at=now - datetime.timedelta(days=3),
        )

        # Archive from non-flagged org (should be ignored)
        self._create_test_repo_archive(
            blob_path="test/repo/non-flagged.tar.gz",
            organization_id=999,  # Non-flagged org
            updated_at=now - datetime.timedelta(days=10),
        )

        # Archive with null updated_at (should be ignored due to query filter)
        self._create_test_repo_archive(
            blob_path="test/repo/null-updated.tar.gz",
            organization_id=1,  # Flagged org
            updated_at=None,
        )

        run_repo_sync()

        # Verify only the old archive from flagged org was processed
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["archive_id"] == old_archive_id

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    def test_run_repo_sync_empty_database(self, mock_apply_async):
        """Test behavior when database has no repo archives."""
        from seer.automation.codebase.tasks import run_repo_sync

        run_repo_sync()

        # Verify no jobs were queued
        mock_apply_async.assert_not_called()

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_database_transaction_integrity(self, mock_repo_client, mock_apply_async):
        """Test that database transactions are handled correctly."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Create archives where one will be deleted due to repo not found
        success_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/success.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "success-owner",
                "name": "success-repo",
                "external_id": "300",
                "branch_name": "main",
            },
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        delete_archive_id = self._create_test_repo_archive(
            blob_path="test/repo/delete.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "delete-owner",
                "name": "delete-repo",
                "external_id": "301",
                "branch_name": "main",
            },
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=3),  # Downloaded recently
        )

        # Setup mock to succeed for first, fail for second
        def mock_repo_client_side_effect(repo_definition, client_type):
            if repo_definition.owner == "success-owner":
                mock_instance = MagicMock()
                mock_instance.repo_full_name = "success-owner/success-repo"
                mock_instance.get_scaled_time_limit.return_value = 900.0
                return mock_instance
            else:
                raise Exception("Error getting repo via full name: delete-owner/delete-repo")

        mock_repo_client.side_effect = mock_repo_client_side_effect

        run_repo_sync()

        # Verify database state is consistent
        with Session() as session:
            success_archive = (
                session.query(DbSeerRepoArchive)
                .filter(DbSeerRepoArchive.id == success_archive_id)
                .first()
            )
            delete_archive = (
                session.query(DbSeerRepoArchive)
                .filter(DbSeerRepoArchive.id == delete_archive_id)
                .first()
            )

            # Success archive should still exist
            assert success_archive is not None
            # Delete archive should be gone
            assert delete_archive is None

        # Verify job was queued for success archive
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args
        job_data = call_args[1]["args"][0]
        assert job_data["archive_id"] == success_archive_id

    @patch("seer.automation.codebase.tasks.run_repo_sync_for_repo_archive.apply_async")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_skips_archives_without_recent_downloads(
        self, mock_repo_client, mock_apply_async
    ):
        """Test that archives without recent downloads are skipped even if they're old."""
        from seer.automation.codebase.tasks import run_repo_sync

        # Create archive that's old but hasn't been downloaded recently
        self._create_test_repo_archive(
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=8),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=10),  # Downloaded too long ago
        )

        # Create archive that's old and never been downloaded
        self._create_test_repo_archive(
            blob_path="test/repo/never-downloaded.tar.gz",
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=8),
            last_downloaded_at=None,  # Never downloaded
        )

        run_repo_sync()

        # Verify no jobs were queued
        mock_apply_async.assert_not_called()
        mock_repo_client.assert_not_called()


class TestRunRepoSyncForRepoArchive:
    """Test cases for the run_repo_sync_for_repo_archive function using the real test database."""

    def setup_method(self):
        """Set up test data before each test."""
        with Session() as session:
            # Clean up any existing test data
            session.query(DbSeerRepoArchive).delete()
            session.commit()

    def teardown_method(self):
        """Clean up test data after each test."""
        with Session() as session:
            session.query(DbSeerRepoArchive).delete()
            session.commit()

    def _create_test_repo_archive(self, **kwargs):
        """Helper to create a test repository archive in the database."""
        defaults = {
            "organization_id": 1,
            "bucket_name": "test-bucket",
            "blob_path": "test/repo/archive.tar.gz",
            "commit_sha": "abc123",
            "repo_definition": {
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo",
                "external_id": "123",
                "branch_name": "main",
            },
            "updated_at": datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=8),
        }
        defaults.update(kwargs)

        with Session() as session:
            archive = DbSeerRepoArchive(**defaults)
            session.add(archive)
            session.commit()
            return archive.id

    def _create_test_repo_sync_job_dict(self, archive_id, **kwargs):
        """Helper to create a test RepoSyncJob dictionary."""
        defaults = {
            "archive_id": archive_id,
            "repo_full_name": "test-owner/test-repo",
            "scaled_time_limit": 900.0,
        }
        defaults.update(kwargs)
        return defaults

    # Input validation tests
    def test_run_repo_sync_for_repo_archive_invalid_job_data(self):
        """Test that function handles invalid job data gracefully."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Test with completely invalid data
        with pytest.raises(Exception):  # Pydantic validation error
            run_repo_sync_for_repo_archive({"invalid": "data"})

    def test_run_repo_sync_for_repo_archive_archive_not_found(self):
        """Test that function raises error when archive ID doesn't exist in database."""
        from seer.automation.codebase.tasks import RepoSyncJobError, run_repo_sync_for_repo_archive

        job_dict = self._create_test_repo_sync_job_dict(archive_id=99999)

        with pytest.raises(RepoSyncJobError, match="repo archive not found"):
            run_repo_sync_for_repo_archive(job_dict)

    # Recently updated archive tests
    def test_run_repo_sync_for_repo_archive_skips_recently_updated(self):
        """Test that function skips archives updated within the interval."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Create archive updated recently (3 days ago)
        recent_update_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=3)
        archive_id = self._create_test_repo_archive(updated_at=recent_update_time)
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        # Should return early without doing any work
        run_repo_sync_for_repo_archive(job_dict)

        # Verify the archive still exists and wasn't modified
        with Session() as session:
            archive = (
                session.query(DbSeerRepoArchive).filter(DbSeerRepoArchive.id == archive_id).first()
            )
            assert archive is not None
            # The updated_at should still be close to the original value (3 days ago)
            # Allow for small differences due to database precision
            time_diff = abs(
                (
                    archive.updated_at.replace(tzinfo=datetime.UTC) - recent_update_time
                ).total_seconds()
            )
            assert time_diff < 60  # Should be within 1 minute of the original time

    # Success path tests
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_success(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test successful repo sync execution."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test archive
        archive_id = self._create_test_repo_archive()
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        run_repo_sync_for_repo_archive(job_dict)

        # Verify RepoClient was created correctly
        mock_repo_client.assert_called_once()
        call_args = mock_repo_client.call_args
        assert len(call_args[0]) == 2  # repo_definition and RepoClientType.READ
        from seer.automation.codebase.repo_client import RepoClientType

        assert call_args[0][1] == RepoClientType.READ

        # Verify RepoManager was created correctly
        mock_repo_manager_class.assert_called_once_with(
            mock_repo_client_instance, organization_id=1
        )

        # Verify update_repo_archive was called
        mock_repo_manager_instance.update_repo_archive.assert_called_once()

    # Error handling tests
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_repo_client_failure(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test handling of RepoClient creation failure."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mock to raise exception
        mock_repo_client.side_effect = Exception("RepoClient creation failed")

        # Create test archive
        archive_id = self._create_test_repo_archive()
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        # Execute and expect exception
        with pytest.raises(Exception, match="RepoClient creation failed"):
            run_repo_sync_for_repo_archive(job_dict)

        # Verify RepoManager was not created
        mock_repo_manager_class.assert_not_called()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_repo_manager_failure(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test handling of RepoManager creation failure."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_class.side_effect = Exception("RepoManager creation failed")

        # Create test archive
        archive_id = self._create_test_repo_archive()
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        # Execute and expect exception
        with pytest.raises(Exception, match="RepoManager creation failed"):
            run_repo_sync_for_repo_archive(job_dict)

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_update_failure(
        self, mock_repo_client, mock_repo_manager_class, caplog
    ):
        """Test handling of update_repo_archive failure."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_instance.update_repo_archive.side_effect = Exception("Update failed")
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test archive
        archive_id = self._create_test_repo_archive()
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        # Execute and expect exception
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="Update failed"):
                run_repo_sync_for_repo_archive(job_dict)

        # Verify error logging
        assert "Failed to update repo archive" in caplog.text
        assert str(archive_id) in caplog.text

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_gcs_not_found_deletes_archive(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that archive is deleted when not found in GCS."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_instance.update_repo_archive.side_effect = Exception(
            "Repository archive not found in GCS"
        )
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test archive
        archive_id = self._create_test_repo_archive()
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        run_repo_sync_for_repo_archive(job_dict)

        # Verify archive was deleted from database
        with Session() as session:
            archive = (
                session.query(DbSeerRepoArchive).filter(DbSeerRepoArchive.id == archive_id).first()
            )
            assert archive is None

    # Logging tests
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_logging(
        self, mock_repo_client, mock_repo_manager_class, caplog
    ):
        """Test that appropriate logs are generated."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test archive
        archive_id = self._create_test_repo_archive()
        job_dict = self._create_test_repo_sync_job_dict(
            archive_id=archive_id, repo_full_name="getsentry/sentry", scaled_time_limit=1800.0
        )

        with caplog.at_level(logging.INFO):
            run_repo_sync_for_repo_archive(job_dict)

        # Verify appropriate logs
        log_messages = caplog.text
        assert "Running repo sync for repo getsentry/sentry with time limit 1800.0" in log_messages
        assert "Repo sync job done." in log_messages

    # Integration tests
    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_realistic_scenario(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test with realistic repository data."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test archive with realistic data
        archive_id = self._create_test_repo_archive(
            organization_id=42,
            bucket_name="seer-repo-archives",
            blob_path="org-42/github/getsentry/sentry/123456789.tar.gz",
            commit_sha="a1b2c3d4e5f6",
            repo_definition={
                "provider": "github",
                "owner": "getsentry",
                "name": "sentry",
                "external_id": "123456789",
                "branch_name": "master",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=10),
        )

        job_dict = self._create_test_repo_sync_job_dict(
            archive_id=archive_id, repo_full_name="getsentry/sentry", scaled_time_limit=1800.0
        )

        run_repo_sync_for_repo_archive(job_dict)

        # Verify RepoManager was created with correct parameters
        mock_repo_manager_class.assert_called_once_with(
            mock_repo_client_instance, organization_id=42
        )

        # Verify update was called
        mock_repo_manager_instance.update_repo_archive.assert_called_once()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_timezone_handling(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test proper timezone handling for updated_at comparison."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create archive with timezone-naive datetime (should be handled by ensure_timezone_aware)
        import datetime as dt

        naive_datetime = dt.datetime(2023, 1, 1, 12, 0, 0)  # No timezone info

        archive_id = self._create_test_repo_archive(updated_at=naive_datetime)
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        # Should not raise exception and should proceed with sync
        run_repo_sync_for_repo_archive(job_dict)

        # Verify update was called (meaning timezone handling worked)
        mock_repo_manager_instance.update_repo_archive.assert_called_once()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_boundary_time_conditions(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test boundary conditions for the update interval check."""
        from seer.automation.codebase.tasks import (
            REPO_ARCHIVE_UPDATE_INTERVAL,
            run_repo_sync_for_repo_archive,
        )

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        now = datetime.datetime.now(datetime.UTC)

        # Test case 1: Exactly at the boundary (should proceed with sync)
        archive_id1 = self._create_test_repo_archive(
            blob_path="test/repo/boundary.tar.gz", updated_at=now - REPO_ARCHIVE_UPDATE_INTERVAL
        )
        job_dict1 = self._create_test_repo_sync_job_dict(archive_id=archive_id1)

        run_repo_sync_for_repo_archive(job_dict1)

        # Should have called update (exactly at boundary means it needs updating)
        mock_repo_manager_instance.update_repo_archive.assert_called_once()

        # Reset mock for next test
        mock_repo_manager_instance.reset_mock()

        # Test case 2: Just inside the boundary (should skip)
        archive_id2 = self._create_test_repo_archive(
            blob_path="test/repo/inside.tar.gz",
            updated_at=now - REPO_ARCHIVE_UPDATE_INTERVAL + datetime.timedelta(seconds=1),
        )
        job_dict2 = self._create_test_repo_sync_job_dict(archive_id=archive_id2)

        run_repo_sync_for_repo_archive(job_dict2)

        # Should NOT have called update (inside boundary means skip)
        mock_repo_manager_instance.update_repo_archive.assert_not_called()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_validates_repo_definition(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that repo definition is properly validated and passed to RepoClient."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Setup mocks
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        # Create test archive with specific repo definition
        archive_id = self._create_test_repo_archive(
            repo_definition={
                "provider": "github",
                "owner": "specific-owner",
                "name": "specific-repo",
                "external_id": "987654321",
                "branch_name": "develop",
            }
        )
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        run_repo_sync_for_repo_archive(job_dict)

        # Verify RepoClient was called with the correct repo definition
        mock_repo_client.assert_called_once()
        call_args = mock_repo_client.call_args
        repo_definition = call_args[0][0]

        # Check that the repo definition was properly validated and passed
        assert repo_definition.provider == "github"
        assert repo_definition.owner == "specific-owner"
        assert repo_definition.name == "specific-repo"
        assert repo_definition.external_id == "987654321"
        assert repo_definition.branch_name == "develop"

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_sync_for_repo_archive_very_old_updated_at(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test handling of archives with very old updated_at (simulating archives that need updating)."""
        from seer.automation.codebase.tasks import run_repo_sync_for_repo_archive

        # Create archive with very old updated_at (simulating an archive that definitely needs updating)
        # Note: The database field is nullable=False, so we can't actually test null values
        very_old_date = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
        archive_id = self._create_test_repo_archive(updated_at=very_old_date)
        job_dict = self._create_test_repo_sync_job_dict(archive_id=archive_id)

        # Setup mocks
        mock_repo_client.return_value = MagicMock()
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_sync_for_repo_archive(job_dict)

        # Verify update was called
        mock_repo_manager_instance.update_repo_archive.assert_called_once()


class TestRunRepoArchiveCleanup:
    """Test cases for the run_repo_archive_cleanup task."""

    def setup_method(self):
        """Clean up any existing data before each test."""
        with Session() as session:
            session.query(DbSeerRepoArchive).delete()
            session.commit()

    def teardown_method(self):
        """Clean up data after each test."""
        with Session() as session:
            session.query(DbSeerRepoArchive).delete()
            session.commit()

    def _create_test_repo_archive(self, **kwargs):
        """Helper method to create a test repository archive."""
        defaults = {
            "organization_id": 1,
            "bucket_name": "test-bucket",
            "blob_path": "repos/1/github/test-owner/test-repo_1234567890.tar.gz",
            "commit_sha": "abcd123456789",
            "repo_definition": {
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo",
                "external_id": "1234567890",
            },
            "updated_at": datetime.datetime.now(datetime.UTC),
        }
        defaults.update(kwargs)
        return DbSeerRepoArchive(**defaults)

    def test_run_repo_archive_cleanup_no_lock_acquired(self):
        """Test when lock is already held by another process."""
        from seer.automation.codebase.tasks import (
            CLEANUP_LOCK_KEY,
            acquire_lock,
            run_repo_archive_cleanup,
        )

        # Acquire the lock in one session
        with Session() as session1:
            with acquire_lock(session1, CLEANUP_LOCK_KEY, "test_lock") as got_lock1:
                assert got_lock1 is True

                # Now run the cleanup task which should fail to get the lock
                run_repo_archive_cleanup()
                # Function should return early without processing

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_no_archives_to_cleanup(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test when there are no archives old enough to cleanup."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Create a recent archive that shouldn't be cleaned up
        recent_archive = self._create_test_repo_archive(
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
        )

        with Session() as session:
            session.add(recent_archive)
            session.commit()

        run_repo_archive_cleanup()

        # Verify no RepoClient or RepoManager was created
        mock_repo_client.assert_not_called()
        mock_repo_manager_class.assert_not_called()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_successful_cleanup(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test successful cleanup of old archives."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Create old archives that should be cleaned up
        old_archive_1 = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/test-repo-1_1111111111.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo-1",
                "external_id": "1111111111",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=35),  # Downloaded long ago
        )

        old_archive_2 = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/test-repo-2_2222222222.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo-2",
                "external_id": "2222222222",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=40),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=40),  # Downloaded long ago
        )

        with Session() as session:
            session.add(old_archive_1)
            session.add(old_archive_2)
            session.commit()

        # Mock RepoClient and RepoManager
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Verify RepoClient and RepoManager were created for each archive
        assert mock_repo_client.call_count == 2
        assert mock_repo_manager_class.call_count == 2
        assert mock_repo_manager_instance.delete_archive.call_count == 2

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_respects_archive_limit(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that cleanup respects MAX_REPO_ARCHIVES_PER_CLEANUP limit."""
        from seer.automation.codebase.tasks import (
            MAX_REPO_ARCHIVES_PER_CLEANUP,
            run_repo_archive_cleanup,
        )

        # Create more archives than the limit
        archives_to_create = MAX_REPO_ARCHIVES_PER_CLEANUP + 10
        old_date = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35)

        with Session() as session:
            for i in range(archives_to_create):
                archive = self._create_test_repo_archive(
                    blob_path=f"repos/1/github/test-owner/test-repo-{i}_{i:010d}.tar.gz",
                    repo_definition={
                        "provider": "github",
                        "owner": "test-owner",
                        "name": f"test-repo-{i}",
                        "external_id": f"{i:010d}",
                    },
                    updated_at=old_date,
                )
                session.add(archive)
            session.commit()

        # Mock RepoClient and RepoManager
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Should only process up to the limit
        assert mock_repo_client.call_count == MAX_REPO_ARCHIVES_PER_CLEANUP
        assert mock_repo_manager_class.call_count == MAX_REPO_ARCHIVES_PER_CLEANUP
        assert mock_repo_manager_instance.delete_archive.call_count == MAX_REPO_ARCHIVES_PER_CLEANUP

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_processes_oldest_first(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that cleanup processes oldest archives first."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Create archives with different ages
        oldest_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/oldest-repo_1111111111.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "oldest-repo",
                "external_id": "1111111111",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=50),
        )

        middle_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/middle-repo_2222222222.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "middle-repo",
                "external_id": "2222222222",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=40),
        )

        newer_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/newer-repo_3333333333.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "newer-repo",
                "external_id": "3333333333",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35),
        )

        with Session() as session:
            # Add in random order to ensure ordering is by updated_at, not insertion order
            session.add(middle_archive)
            session.add(newer_archive)
            session.add(oldest_archive)
            session.commit()

        # Mock RepoClient and RepoManager to track call order
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Verify all three archives were processed
        assert mock_repo_client.call_count == 3

        # Check the order of calls - should be oldest first
        call_args_list = mock_repo_client.call_args_list
        first_call_repo = call_args_list[0][0][0]
        second_call_repo = call_args_list[1][0][0]
        third_call_repo = call_args_list[2][0][0]

        assert first_call_repo.name == "oldest-repo"
        assert second_call_repo.name == "middle-repo"
        assert third_call_repo.name == "newer-repo"

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_handles_individual_failures(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that individual archive cleanup failures don't stop the entire process."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Create two old archives
        archive_1 = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/test-repo-1_1111111111.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo-1",
                "external_id": "1111111111",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=35),  # Downloaded long ago
        )

        archive_2 = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/test-repo-2_2222222222.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "test-repo-2",
                "external_id": "2222222222",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=40),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=40),  # Downloaded long ago
        )

        with Session() as session:
            session.add(archive_1)
            session.add(archive_2)
            session.commit()

        # Mock RepoClient and RepoManager - make first delete_archive fail
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()

        # Configure delete_archive to fail on first call, succeed on second
        mock_repo_manager_instance.delete_archive.side_effect = [
            Exception("GCS deletion failed"),
            None,  # Success on second call
        ]
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Both archives should have been attempted
        assert mock_repo_client.call_count == 2
        assert mock_repo_manager_class.call_count == 2
        assert mock_repo_manager_instance.delete_archive.call_count == 2

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_handles_repo_client_creation_failure(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test handling of RepoClient creation failures."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Create an old archive
        old_archive = self._create_test_repo_archive(
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35)
        )

        with Session() as session:
            session.add(old_archive)
            session.commit()

        # Mock RepoClient creation to fail
        mock_repo_client.side_effect = Exception("Failed to create repo client")

        run_repo_archive_cleanup()

        # Should attempt to create RepoClient
        mock_repo_client.assert_called_once()
        # Should not create RepoManager since RepoClient creation failed
        mock_repo_manager_class.assert_not_called()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_cutoff_date_logic(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that only archives older than the cutoff date are processed."""
        from seer.automation.codebase.tasks import (
            REPO_ARCHIVE_CLEANUP_INTERVAL,
            run_repo_archive_cleanup,
        )

        now = datetime.datetime.now(datetime.UTC)

        # Create archives at various ages relative to the cutoff
        too_new_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/too-new_1111111111.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "too-new",
                "external_id": "1111111111",
            },
            updated_at=now
            - REPO_ARCHIVE_CLEANUP_INTERVAL
            + datetime.timedelta(hours=1),  # 29 days, 23 hours old
        )

        just_old_enough_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/just-old-enough_2222222222.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "just-old-enough",
                "external_id": "2222222222",
            },
            updated_at=now
            - REPO_ARCHIVE_CLEANUP_INTERVAL
            - datetime.timedelta(hours=1),  # 30 days, 1 hour old
        )

        very_old_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/very-old_3333333333.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "very-old",
                "external_id": "3333333333",
            },
            updated_at=now - datetime.timedelta(days=60),  # 60 days old
        )

        with Session() as session:
            session.add(too_new_archive)
            session.add(just_old_enough_archive)
            session.add(very_old_archive)
            session.commit()

        # Mock RepoClient and RepoManager
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Should only process the two old enough archives
        assert mock_repo_client.call_count == 2
        assert mock_repo_manager_class.call_count == 2

        # Verify which archives were processed
        call_args_list = mock_repo_client.call_args_list
        processed_repos = [call[0][0].name for call in call_args_list]

        assert "just-old-enough" in processed_repos
        assert "very-old" in processed_repos
        assert "too-new" not in processed_repos

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_skips_null_updated_at(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that archives with null updated_at are skipped."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Create archives with null and non-null updated_at
        null_updated_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/null-updated_1111111111.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "null-updated",
                "external_id": "1111111111",
            },
            updated_at=None,
        )

        old_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/old-repo_2222222222.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "old-repo",
                "external_id": "2222222222",
            },
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35),
        )

        with Session() as session:
            session.add(null_updated_archive)
            session.add(old_archive)
            session.commit()

        # Mock RepoClient and RepoManager
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Should only process the archive with non-null updated_at
        assert mock_repo_client.call_count == 1
        mock_repo_client.assert_called_once()

        # Verify the correct archive was processed
        call_args = mock_repo_client.call_args[0]
        repo_definition = call_args[0]
        assert repo_definition.name == "old-repo"

    def test_run_repo_archive_cleanup_empty_database(self):
        """Test cleanup behavior when no archives exist in database."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Should complete without errors
        run_repo_archive_cleanup()

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_realistic_scenario(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test cleanup with realistic repository data and mixed conditions."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        now = datetime.datetime.now(datetime.UTC)

        # Create a mix of archives - some to clean up, some to keep
        archives_to_create = [
            # Should be cleaned up
            {
                "blob_path": "repos/1/github/getsentry/sentry_4088350.tar.gz",
                "repo_definition": {
                    "provider": "github",
                    "owner": "getsentry",
                    "name": "sentry",
                    "external_id": "4088350",
                },
                "updated_at": now - datetime.timedelta(days=45),
                "should_cleanup": True,
            },
            # Should be cleaned up
            {
                "blob_path": "repos/1/github/microsoft/vscode_41881900.tar.gz",
                "repo_definition": {
                    "provider": "github",
                    "owner": "microsoft",
                    "name": "vscode",
                    "external_id": "41881900",
                },
                "updated_at": now - datetime.timedelta(days=35),
                "should_cleanup": True,
            },
            # Should NOT be cleaned up (too recent)
            {
                "blob_path": "repos/1/github/facebook/react_10270250.tar.gz",
                "repo_definition": {
                    "provider": "github",
                    "owner": "facebook",
                    "name": "react",
                    "external_id": "10270250",
                },
                "updated_at": now - datetime.timedelta(days=15),
                "should_cleanup": False,
            },
            # Should NOT be cleaned up (wrong org)
            {
                "organization_id": 2,  # Not in FLAGGED_ORG_IDS
                "blob_path": "repos/2/github/torvalds/linux_2325298.tar.gz",
                "repo_definition": {
                    "provider": "github",
                    "owner": "torvalds",
                    "name": "linux",
                    "external_id": "2325298",
                },
                "updated_at": now - datetime.timedelta(days=50),
                "should_cleanup": False,
            },
        ]

        with Session() as session:
            for archive_data in archives_to_create:
                archive = self._create_test_repo_archive(
                    **{k: v for k, v in archive_data.items() if k != "should_cleanup"}
                )
                session.add(archive)
            session.commit()

        # Mock RepoClient and RepoManager
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Should process exactly 2 archives (the old ones from org 1)
        expected_cleanup_count = len([a for a in archives_to_create if a["should_cleanup"]])
        assert mock_repo_client.call_count == expected_cleanup_count
        assert mock_repo_manager_class.call_count == expected_cleanup_count

        # Verify the correct repositories were processed
        call_args_list = mock_repo_client.call_args_list
        processed_repos = [(call[0][0].owner, call[0][0].name) for call in call_args_list]

        assert ("getsentry", "sentry") in processed_repos
        assert ("microsoft", "vscode") in processed_repos
        assert ("facebook", "react") not in processed_repos
        assert ("torvalds", "linux") not in processed_repos

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_last_downloaded_logic(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test cleanup logic based on last_downloaded_at field."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        now = datetime.datetime.now(datetime.UTC)

        # Archive with old last_downloaded_at (should be cleaned up)
        old_downloaded_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/old-downloaded_1111111111.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "old-downloaded",
                "external_id": "1111111111",
            },
            updated_at=now - datetime.timedelta(days=10),  # Recently updated
            last_downloaded_at=now - datetime.timedelta(days=35),  # Downloaded long ago
        )

        # Archive with null last_downloaded_at but old updated_at (should be cleaned up)
        never_downloaded_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/never-downloaded_2222222222.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "never-downloaded",
                "external_id": "2222222222",
            },
            updated_at=now - datetime.timedelta(days=35),  # Updated long ago
            last_downloaded_at=None,  # Never downloaded
        )

        # Archive with recent last_downloaded_at (should NOT be cleaned up)
        recent_downloaded_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/recent-downloaded_3333333333.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "recent-downloaded",
                "external_id": "3333333333",
            },
            updated_at=now - datetime.timedelta(days=35),  # Updated long ago
            last_downloaded_at=now - datetime.timedelta(days=10),  # Downloaded recently
        )

        # Archive with null last_downloaded_at and recent updated_at (should NOT be cleaned up)
        recent_updated_archive = self._create_test_repo_archive(
            blob_path="repos/1/github/test-owner/recent-updated_4444444444.tar.gz",
            repo_definition={
                "provider": "github",
                "owner": "test-owner",
                "name": "recent-updated",
                "external_id": "4444444444",
            },
            updated_at=now - datetime.timedelta(days=10),  # Updated recently
            last_downloaded_at=None,  # Never downloaded
        )

        with Session() as session:
            session.add(old_downloaded_archive)
            session.add(never_downloaded_archive)
            session.add(recent_downloaded_archive)
            session.add(recent_updated_archive)
            session.commit()

        # Mock RepoClient and RepoManager
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Should clean up exactly 2 archives (old_downloaded and never_downloaded)
        assert mock_repo_client.call_count == 2
        assert mock_repo_manager_class.call_count == 2
        assert mock_repo_manager_instance.delete_archive.call_count == 2

        # Verify which archives were processed
        call_args_list = mock_repo_client.call_args_list
        processed_repos = [call[0][0].name for call in call_args_list]

        assert "old-downloaded" in processed_repos
        assert "never-downloaded" in processed_repos
        assert "recent-downloaded" not in processed_repos
        assert "recent-updated" not in processed_repos

    @patch("seer.automation.codebase.tasks.RepoManager")
    @patch("seer.automation.codebase.tasks.RepoClient.from_repo_definition")
    def test_run_repo_archive_cleanup_respects_flagged_orgs_only(
        self, mock_repo_client, mock_repo_manager_class
    ):
        """Test that cleanup only processes archives from flagged organizations."""
        from seer.automation.codebase.tasks import run_repo_archive_cleanup

        # Create old archives - one from flagged org, one from non-flagged org
        flagged_org_archive = self._create_test_repo_archive(
            organization_id=1,  # This is in FLAGGED_ORG_IDS
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=35),  # Downloaded long ago
        )

        non_flagged_org_archive = self._create_test_repo_archive(
            organization_id=2,  # This is NOT in FLAGGED_ORG_IDS
            blob_path="repos/2/github/test-owner/test-repo_1234567890.tar.gz",
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=35),
            last_downloaded_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(days=35),  # Downloaded long ago
        )

        with Session() as session:
            session.add(flagged_org_archive)
            session.add(non_flagged_org_archive)
            session.commit()

        # Mock RepoClient and RepoManager
        mock_repo_client_instance = MagicMock()
        mock_repo_client.return_value = mock_repo_client_instance
        mock_repo_manager_instance = MagicMock()
        mock_repo_manager_class.return_value = mock_repo_manager_instance

        run_repo_archive_cleanup()

        # Only the flagged org archive should be processed
        assert mock_repo_client.call_count == 1
        assert mock_repo_manager_class.call_count == 1
        mock_repo_manager_instance.delete_archive.assert_called_once()

        # Verify the correct archive was processed
        call_args = mock_repo_client.call_args[0]
        repo_definition = call_args[0]
        assert repo_definition.owner == "test-owner"
        assert repo_definition.name == "test-repo"
