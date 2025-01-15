import json

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text

# Import the migration to test
from migrations.versions.cefa0023a864_codebase_changes_tied_to_step import (
    downgrade,
    migrate_old_codebase_changes_to_new,
    upgrade,
)


@pytest.fixture
def db_engine():
    """Create an in-memory SQLite database"""
    engine = create_engine("sqlite:///:memory:")

    # Create the run_state table
    with engine.connect() as conn:
        conn.execute(
            text(
                """
            CREATE TABLE run_state (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """
            )
        )

    return engine


@pytest.fixture
def sample_run_state():
    """Sample data representing an old format run_state"""
    return {
        "codebases": {
            "repo1": {
                "repo_external_id": "repo1",
                "file_changes": [
                    {"change_type": "edit", "path": "test.py", "description": "Test change"}
                ],
            }
        },
        "steps": [
            {"id": "123", "key": "default", "insights": []},
            {
                "id": "124",
                "key": "changes",
                "changes": [
                    {
                        "repo_external_id": "repo1",
                        "repo_name": "Test Repo",
                        "title": "Test Change",
                        "description": "Test Description",
                        "diff": [],
                    }
                ],
            },
        ],
    }


def test_upgrade_migration(db_engine, sample_run_state):
    # Insert test data
    with db_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO run_state (id, value) VALUES (:id, :value)"),
            {"id": 1, "value": json.dumps(sample_run_state)},
        )
        conn.commit()

    # Run upgrade
    with db_engine.begin() as conn:
        context = MigrationContext.configure(
            conn,
            opts={
                "compare_type": True,
                "compare_server_default": True,
            },
        )
        with Operations.context(context):
            upgrade()

        result = conn.execute(text("SELECT value FROM run_state WHERE id = 1")).scalar()
        data = json.loads(result)

        # Assert the new structure exists
        changes_step = next(step for step in data["steps"] if step["key"] == "changes")
        assert "codebase_changes" in changes_step
        assert "repo1" in changes_step["codebase_changes"]

        assert "codebases" in data  # Verify codebases is maintained
        assert "repo1" in data["codebases"]

        # Verify the migrated data
        migrated = changes_step["codebase_changes"]["repo1"]
        assert migrated["repo_name"] == "Test Repo"
        assert migrated["details"]["title"] == "Test Change"
        assert len(migrated["file_changes"]) == 1


def test_downgrade_migration(db_engine, sample_run_state):
    # First upgrade
    with db_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO run_state (id, value) VALUES (:id, :value)"),
            {"id": 1, "value": json.dumps(sample_run_state)},
        )
        conn.commit()

    # Run upgrade then downgrade
    with db_engine.begin() as conn:
        context = MigrationContext.configure(
            conn,
            opts={
                "compare_type": True,
                "compare_server_default": True,
            },
        )
        with Operations.context(context):
            upgrade()
            downgrade()

        # Verify the changes were reverted
        result = conn.execute(text("SELECT value FROM run_state WHERE id = 1")).scalar()
        data = json.loads(result)

        changes_step = next(step for step in data["steps"] if step["key"] == "changes")
        assert "codebase_changes" not in changes_step

        # Verify codebases data is restored
        assert "codebases" in data
        assert "repo1" in data["codebases"]
        assert data["codebases"]["repo1"]["repo_external_id"] == "repo1"
        assert len(data["codebases"]["repo1"]["file_changes"]) == 1
        assert data["codebases"]["repo1"]["file_changes"][0]["change_type"] == "edit"
        assert data["codebases"]["repo1"]["file_changes"][0]["path"] == "test.py"


def test_migrate_old_codebase_changes_to_new():
    """Test the migration helper function directly"""
    codebase_states = {
        "repo1": {
            "repo_external_id": "repo1",
            "file_changes": [
                {"change_type": "edit", "path": "test.py", "description": "Test change"}
            ],
        }
    }

    changes = [
        {
            "repo_external_id": "repo1",
            "repo_name": "Test Repo",
            "title": "Test Change",
            "description": "Test Description",
            "diff": [],
        }
    ]

    result = migrate_old_codebase_changes_to_new(codebase_states, changes)

    assert "repo1" in result
    assert result["repo1"]["repo_name"] == "Test Repo"
    assert result["repo1"]["details"]["title"] == "Test Change"
    assert len(result["repo1"]["file_changes"]) == 1
