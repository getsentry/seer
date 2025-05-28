from pytest_alembic import MigrationContext
from sqlalchemy import insert

from seer.db import DbRunState, Session


def test_run_state_migration(alembic_runner: MigrationContext):
    alembic_runner.migrate_up_to("4a87188db505")

    with Session() as session:
        for id, json in (
            (1, {"a": "\x00"}),  # Valid, but pg sql won't decode it well
            (2, {}),  # Valid, missing keys
            (3, {"updated_at": None}),  # Valid, key is null
            (4, {"last_triggered_at": None}),  # Valid, key is null
            (5, {"updated_at": "2020-06-15T13:45:30"}),  # Valid, stuff
            (6, {"last_triggered_at": "2019-04-12T11:15:31"}),  # Valid, stuff
        ):
            session.execute(
                insert(alembic_runner.table_at_revision("run_state")).values(id=id, value=json)
            )
        session.commit()

    alembic_runner.migrate_up_to("head")

    with Session() as session:
        for i in range(1, 7):
            state = session.query(DbRunState).get(i)
            assert state.created_at
            assert state.last_triggered_at
            assert state.updated_at
