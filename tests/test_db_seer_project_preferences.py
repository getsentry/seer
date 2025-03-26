from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError

from seer.db import DbSeerProjectPreference, Session


class TestDbSeerProjectPreference:
    def setup_method(self):
        """Setup method that runs before each test"""
        # Clean up any existing test records to ensure test isolation
        with Session() as session:
            session.query(DbSeerProjectPreference).filter(
                DbSeerProjectPreference.project_id.in_([123, 456, 789])
            ).delete()
            session.commit()

    def teardown_method(self):
        """Teardown method that runs after each test"""
        # Clean up test records
        with Session() as session:
            session.query(DbSeerProjectPreference).filter(
                DbSeerProjectPreference.project_id.in_([123, 456, 789])
            ).delete()
            session.commit()

    def test_db_seer_project_preference_create_and_read(self):
        """Test creating and retrieving a DbSeerProjectPreference"""
        # Create a new preference
        with Session() as session:
            preference = DbSeerProjectPreference(
                project_id=123,
                organization_id=456,
                repositories=[
                    {
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "test-id",
                        "provider": "github",
                        "branch_name": "main",
                        "instructions": "# Test instructions",
                    }
                ],
            )
            session.add(preference)
            session.commit()

        # Retrieve and verify
        with Session() as session:
            retrieved = session.get(DbSeerProjectPreference, 123)
            assert retrieved is not None
            assert retrieved.project_id == 123
            assert retrieved.organization_id == 456
            assert len(retrieved.repositories) == 1
            assert retrieved.repositories[0]["owner"] == "test-owner"
            assert retrieved.repositories[0]["name"] == "test-repo"
            assert retrieved.repositories[0]["provider"] == "github"
            assert retrieved.repositories[0]["branch_name"] == "main"
            assert retrieved.repositories[0]["instructions"] == "# Test instructions"

    def test_db_seer_project_preference_unique_constraint(self):
        """Test the unique constraint on organization_id and project_id"""
        # First add should succeed
        with Session() as session:
            preference1 = DbSeerProjectPreference(
                project_id=123,
                organization_id=456,
                repositories=[],
            )
            session.add(preference1)
            session.commit()

        # Second add with same project_id and organization_id should fail
        with pytest.raises(IntegrityError):
            with Session() as session:
                preference2 = DbSeerProjectPreference(
                    project_id=123,
                    organization_id=456,
                    repositories=[],
                )
                session.add(preference2)
                session.commit()

    def test_db_seer_project_preference_update(self):
        """Test updating a DbSeerProjectPreference"""
        # Create a preference first
        with Session() as session:
            preference = DbSeerProjectPreference(
                project_id=123,
                organization_id=456,
                repositories=[
                    {
                        "owner": "test-owner",
                        "name": "test-repo",
                        "external_id": "test-id",
                        "provider": "github",
                    }
                ],
            )
            session.add(preference)
            session.commit()

        # Update the preference
        with Session() as session:
            preference = session.get(DbSeerProjectPreference, 123)
            preference.repositories = [
                {
                    "owner": "test-owner",
                    "name": "test-repo",
                    "external_id": "test-id",
                    "provider": "github",
                    "branch_name": "feature-branch",
                    "instructions": "# Updated instructions",
                }
            ]
            session.commit()

        # Verify the update
        with Session() as session:
            updated = session.get(DbSeerProjectPreference, 123)
            assert len(updated.repositories) == 1
            assert updated.repositories[0]["branch_name"] == "feature-branch"
            assert updated.repositories[0]["instructions"] == "# Updated instructions"

    def test_db_seer_project_preference_delete(self):
        """Test deleting a DbSeerProjectPreference"""
        # Create a preference first
        with Session() as session:
            preference = DbSeerProjectPreference(
                project_id=789,
                organization_id=456,
                repositories=[],
            )
            session.add(preference)
            session.commit()

        # Verify it exists
        with Session() as session:
            retrieved = session.get(DbSeerProjectPreference, 789)
            assert retrieved is not None

        # Delete the preference
        with Session() as session:
            preference = session.get(DbSeerProjectPreference, 789)
            session.delete(preference)
            session.commit()

        # Verify it's gone
        with Session() as session:
            deleted = session.get(DbSeerProjectPreference, 789)
            assert deleted is None

    def test_db_seer_project_preference_query(self):
        """Test querying for DbSeerProjectPreference by organization_id"""
        # Create preferences with different organization_ids
        with Session() as session:
            preference1 = DbSeerProjectPreference(
                project_id=123,
                organization_id=456,
                repositories=[],
            )
            preference2 = DbSeerProjectPreference(
                project_id=789,
                organization_id=456,  # Same org_id as preference1
                repositories=[],
            )
            session.add_all([preference1, preference2])
            session.commit()

        # Query by organization_id
        with Session() as session:
            preferences = (
                session.query(DbSeerProjectPreference)
                .filter(DbSeerProjectPreference.organization_id == 456)
                .all()
            )

            # Verify we got both preferences for this organization
            assert len(preferences) == 2
            project_ids = [p.project_id for p in preferences]
            assert 123 in project_ids
            assert 789 in project_ids
