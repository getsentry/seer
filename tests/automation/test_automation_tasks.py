import datetime

from seer.automation.tasks import delete_all_runs_before, delete_old_automation_runs
from seer.db import DbRunState, Session


class TestDeleteOldAutofixRuns:
    def test_old_data_is_deleted(self):
        old_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
            value="test_value",
        )
        with Session() as session:
            session.add(old_run)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_count = delete_all_runs_before(before_date)

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == old_run.id).first()
        assert remaining_run is None
        assert deleted_count == 1

    def test_data_not_deleted_within_ttl(self):
        recent_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=80),
            value="test_value",
        )
        with Session() as session:
            session.add(recent_run)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_count = delete_all_runs_before(before_date)

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == recent_run.id).first()
        assert remaining_run is not None
        assert remaining_run.last_triggered_at > before_date
        assert deleted_count == 0

    def test_batch_delete(self):
        old_runs = [
            DbRunState(
                last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
                value="test_value",
            )
            for _ in range(25)
        ]
        with Session() as session:
            session.bulk_save_objects(old_runs)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_count = delete_all_runs_before(before_date, batch_size=10)

        with Session() as session:
            remaining_runs = (
                session.query(DbRunState).filter(DbRunState.last_triggered_at < before_date).count()
            )
        assert remaining_runs == 0
        assert deleted_count == 25

    def test_delete_old_automation_runs_task(self):
        old_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
            value="test_value",
        )
        with Session() as session:
            session.add(old_run)
            session.commit()

        # Celery task
        delete_old_automation_runs()

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == old_run.id).first()
        assert remaining_run is None
