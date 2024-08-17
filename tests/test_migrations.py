from pytest_alembic import MigrationContext
from sqlalchemy import insert, text

from seer.db import DbRunState, Session


def test_run_state_migration(alembic_runner: MigrationContext):
    def f():
        with alembic_runner.connection_executor.connection.connect() as c:
            return c.execute(
                text(
                    """
            SELECT
        table_schema || '.' || table_name
    FROM
        information_schema.tables
    WHERE
        table_type = 'BASE TABLE'
    AND
        table_schema NOT IN ('pg_catalog', 'information_schema');
            """
                )
            ).all()

    alembic_runner.migrate_up_before("9b8704bd8c4a")

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
            assert state.last_triggered_at
            assert state.updated_at
