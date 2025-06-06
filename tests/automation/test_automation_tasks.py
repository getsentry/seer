import datetime

from seer.automation.tasks import (
    delete_all_runs_before,
    delete_all_summaries_before,
    delete_data_for_ttl,
    delete_expired_blacklist_entries_before,
)
from seer.db import DbIssueSummary, DbLlmRegionBlacklist, DbRunState, Session


class TestDeleteOldAutomationAndIssueSummaryRuns:
    def setup_method(self):
        """Clean up database before each test"""
        with Session() as session:
            session.query(DbRunState).delete()
            session.query(DbIssueSummary).delete()
            session.query(DbLlmRegionBlacklist).delete()
            session.commit()

    def test_old_data_is_deleted(self):
        old_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
            value="test_value",
        )
        old_summary = DbIssueSummary(
            group_id=1,
            summary="summary",
            created_at=datetime.datetime.now() - datetime.timedelta(days=100),
        )
        with Session() as session:
            session.add(old_run)
            session.add(old_summary)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_run_count = delete_all_runs_before(before_date)
        deleted_summary_count = delete_all_summaries_before(before_date)

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == old_run.id).first()
            remaining_summary = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.group_id == old_summary.group_id)
                .first()
            )
        assert remaining_run is None
        assert remaining_summary is None
        assert deleted_run_count == 1
        assert deleted_summary_count == 1

    def test_data_not_deleted_within_ttl(self):
        recent_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=80),
            value="test_value",
        )
        recent_summary = DbIssueSummary(
            group_id=1,
            summary="summary",
            created_at=datetime.datetime.now() - datetime.timedelta(days=80),
        )
        with Session() as session:
            session.add(recent_run)
            session.add(recent_summary)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_run_count = delete_all_runs_before(before_date)
        deleted_summary_count = delete_all_summaries_before(before_date)

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == recent_run.id).first()
            remaining_summary = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.group_id == recent_summary.group_id)
                .first()
            )
        assert remaining_run is not None
        assert remaining_summary is not None
        assert remaining_run.last_triggered_at > before_date
        assert remaining_summary.created_at > before_date
        assert deleted_run_count == 0
        assert deleted_summary_count == 0

    def test_batch_delete(self):
        old_runs = [
            DbRunState(
                last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
                value="test_value",
            )
            for _ in range(25)
        ]
        old_summaries = [
            DbIssueSummary(
                group_id=i,
                summary="summary",
                created_at=datetime.datetime.now() - datetime.timedelta(days=100),
            )
            for i in range(35)
        ]
        with Session() as session:
            session.bulk_save_objects(old_runs)
            session.bulk_save_objects(old_summaries)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_run_count = delete_all_runs_before(before_date, batch_size=10)
        deleted_summary_count = delete_all_summaries_before(before_date, batch_size=10)

        with Session() as session:
            remaining_runs = (
                session.query(DbRunState).filter(DbRunState.last_triggered_at < before_date).count()
            )
            remaining_summaries = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.created_at < before_date)
                .count()
            )
        assert remaining_runs == 0
        assert remaining_summaries == 0
        assert deleted_run_count == 25
        assert deleted_summary_count == 35

    def test_delete_old_autofix_runs_task(self):
        old_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
            value="test_value",
        )
        old_summary = DbIssueSummary(
            group_id=1,
            summary="summary",
            created_at=datetime.datetime.now() - datetime.timedelta(days=100),
        )
        with Session() as session:
            session.add(old_run)
            session.add(old_summary)
            session.commit()

        # Celery task
        delete_data_for_ttl()

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == old_run.id).first()
            remaining_summary = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.group_id == old_summary.group_id)
                .first()
            )
        assert remaining_run is None
        assert remaining_summary is None

    def test_delete_expired_blacklist_entries_before(self):
        """Test that delete_expired_blacklist_entries_before removes only entries that expired before the cutoff date"""
        # Create test data with different expiry dates
        now = datetime.datetime.now(datetime.timezone.utc)
        very_old_time = now - datetime.timedelta(days=100)
        old_time = now - datetime.timedelta(days=50)
        recent_time = now - datetime.timedelta(days=10)

        # Create blacklist entries:
        # 1. Very old entry (should be deleted)
        # 2. Old entry (should be deleted)
        # 3. Recent entry (should NOT be deleted)
        # 4. Active entry (should NOT be deleted)
        with Session() as session:
            very_old_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                blacklisted_at=very_old_time,
                expires_at=very_old_time + datetime.timedelta(minutes=5),
                failure_count=1,
            )
            old_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-west-2",
                blacklisted_at=old_time,
                expires_at=old_time + datetime.timedelta(minutes=5),
                failure_count=1,
            )
            recent_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="eu-west-1",
                blacklisted_at=recent_time,
                expires_at=recent_time + datetime.timedelta(minutes=5),
                failure_count=1,
            )
            active_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="ap-southeast-1",
                blacklisted_at=now,
                expires_at=now + datetime.timedelta(hours=1),  # Still active
                failure_count=1,
            )
            session.add_all([very_old_entry, old_entry, recent_entry, active_entry])
            session.commit()

        # Delete entries that expired before 30 days ago
        cutoff_date = now - datetime.timedelta(days=30)
        deleted_count = delete_expired_blacklist_entries_before(cutoff_date)

        # Verify results
        assert deleted_count == 2  # Should delete very_old_entry and old_entry

        with Session() as session:
            remaining_entries = session.query(DbLlmRegionBlacklist).all()
            remaining_regions = {entry.region for entry in remaining_entries}

            # Should only have recent and active entries left
            assert len(remaining_entries) == 2
            assert remaining_regions == {"eu-west-1", "ap-southeast-1"}

    def test_delete_data_for_ttl_includes_blacklist_cleanup(self):
        """Test that the main TTL task includes blacklist cleanup"""
        # Create old blacklist entry that should be cleaned up
        now = datetime.datetime.now(datetime.timezone.utc)
        old_time = now - datetime.timedelta(days=100)

        with Session() as session:
            old_blacklist_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                blacklisted_at=old_time,
                expires_at=old_time + datetime.timedelta(minutes=5),
                failure_count=1,
            )
            session.add(old_blacklist_entry)
            session.commit()

        # Run the main TTL task
        delete_data_for_ttl()

        # Verify old blacklist entry was cleaned up
        with Session() as session:
            remaining_blacklist_entries = session.query(DbLlmRegionBlacklist).count()
            assert remaining_blacklist_entries == 0
