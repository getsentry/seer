import datetime

from seer.automation.tasks import (
    delete_all_runs_before,
    delete_all_summaries_before,
    delete_data_for_ttl,
)
from seer.db import DbIssueSummary, DbRunState, Session


class TestDeleteOldAutomationAndIssueSummaryRuns:
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
