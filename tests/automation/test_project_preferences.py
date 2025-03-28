import json
import unittest
from unittest import mock

from seer.app import app
from seer.automation.models import RepoDefinition, SeerProjectPreference
from seer.automation.preferences import (
    GetSeerProjectPreferenceRequest,
    GetSeerProjectPreferenceResponse,
    SetSeerProjectPreferenceRequest,
    SetSeerProjectPreferenceResponse,
    get_seer_project_preference,
    set_seer_project_preference,
)
from seer.db import DbSeerProjectPreference


class TestSeerProjectPreferenceModel(unittest.TestCase):
    def test_seer_project_preference_to_db_model(self):
        """Test conversion from SeerProjectPreference to DbSeerProjectPreference"""
        # Create test data
        repos = [
            RepoDefinition(
                owner="test-owner",
                name="test-repo",
                external_id="test-external-id",
                provider="github",
                branch_name="main",
                instructions="# Custom instructions",
            )
        ]
        preference = SeerProjectPreference(
            organization_id=123,
            project_id=456,
            repositories=repos,
        )

        # Convert to DB model
        db_model = preference.to_db_model()

        # Assert conversion is correct
        self.assertEqual(db_model.organization_id, 123)
        self.assertEqual(db_model.project_id, 456)
        self.assertEqual(len(db_model.repositories), 1)
        self.assertEqual(db_model.repositories[0]["owner"], "test-owner")
        self.assertEqual(db_model.repositories[0]["name"], "test-repo")
        self.assertEqual(db_model.repositories[0]["external_id"], "test-external-id")
        self.assertEqual(db_model.repositories[0]["provider"], "github")
        self.assertEqual(db_model.repositories[0]["branch_name"], "main")
        self.assertEqual(db_model.repositories[0]["instructions"], "# Custom instructions")

    def test_seer_project_preference_from_db_model(self):
        """Test conversion from DbSeerProjectPreference to SeerProjectPreference"""
        # Create test DB model
        db_model = DbSeerProjectPreference(
            organization_id=123,
            project_id=456,
            repositories=[
                {
                    "owner": "test-owner",
                    "name": "test-repo",
                    "external_id": "test-external-id",
                    "provider": "github",
                    "branch_name": "feature-branch",
                    "instructions": "# Test instructions",
                }
            ],
        )

        # Convert to Pydantic model
        preference = SeerProjectPreference.from_db_model(db_model)

        # Assert conversion is correct
        self.assertEqual(preference.organization_id, 123)
        self.assertEqual(preference.project_id, 456)
        self.assertEqual(len(preference.repositories), 1)
        self.assertEqual(preference.repositories[0].owner, "test-owner")
        self.assertEqual(preference.repositories[0].name, "test-repo")
        self.assertEqual(preference.repositories[0].external_id, "test-external-id")
        self.assertEqual(preference.repositories[0].provider, "github")
        self.assertEqual(preference.repositories[0].branch_name, "feature-branch")
        self.assertEqual(preference.repositories[0].instructions, "# Test instructions")

    def test_repo_definition_full_name(self):
        """Test the full_name property of RepoDefinition"""
        repo = RepoDefinition(
            owner="test-owner",
            name="test-repo",
            external_id="test-id",
            provider="github",
        )
        self.assertEqual(repo.full_name, "test-owner/test-repo")


class TestSeerProjectPreferenceAPI(unittest.TestCase):
    @mock.patch("seer.automation.preferences.Session")
    def test_get_seer_project_preference_found(self, mock_session):
        """Test get_seer_project_preference when the preference exists"""
        # Setup mock session
        mock_session_instance = mock.MagicMock()
        mock_session.return_value.__enter__.return_value = mock_session_instance

        # Create a mock DB preference
        mock_preference = DbSeerProjectPreference(
            project_id=123,
            organization_id=456,
            repositories=[
                {
                    "owner": "test-owner",
                    "name": "test-repo",
                    "external_id": "test-external-id",
                    "provider": "github",
                }
            ],
        )
        mock_session_instance.get.return_value = mock_preference

        # Call the function
        request = GetSeerProjectPreferenceRequest(project_id=123)
        response = get_seer_project_preference(request)

        # Verify the response
        self.assertIsNotNone(response.preference)
        self.assertEqual(response.preference.project_id, 123)
        self.assertEqual(response.preference.organization_id, 456)
        self.assertEqual(len(response.preference.repositories), 1)
        self.assertEqual(response.preference.repositories[0].owner, "test-owner")
        self.assertEqual(response.preference.repositories[0].provider, "github")

        # Verify the session was used correctly
        mock_session_instance.get.assert_called_once_with(DbSeerProjectPreference, 123)

    @mock.patch("seer.automation.preferences.Session")
    def test_get_seer_project_preference_not_found(self, mock_session):
        """Test get_seer_project_preference when the preference doesn't exist"""
        # Setup mock session
        mock_session_instance = mock.MagicMock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        mock_session_instance.get.return_value = None

        # Call the function
        request = GetSeerProjectPreferenceRequest(project_id=999)
        response = get_seer_project_preference(request)

        # Verify the response
        self.assertIsNone(response.preference)

        # Verify the session was used correctly
        mock_session_instance.get.assert_called_once_with(DbSeerProjectPreference, 999)

    @mock.patch("seer.automation.preferences.Session")
    def test_set_seer_project_preference(self, mock_session):
        """Test set_seer_project_preference functionality"""
        # Setup mock session
        mock_session_instance = mock.MagicMock()
        mock_session.return_value.__enter__.return_value = mock_session_instance

        # Create test data
        repos = [
            RepoDefinition(
                owner="test-owner",
                name="test-repo",
                external_id="test-external-id",
                provider="github",
                branch_name="main",
            )
        ]
        preference = SeerProjectPreference(
            organization_id=123,
            project_id=456,
            repositories=repos,
        )

        # Call the function
        request = SetSeerProjectPreferenceRequest(preference=preference)
        response = set_seer_project_preference(request)

        # Verify the response
        self.assertEqual(response.preference, preference)

        # Verify the session was used correctly
        mock_session_instance.merge.assert_called_once()
        mock_session_instance.commit.assert_called_once()


class TestSeerProjectPreferenceEndpoints(unittest.TestCase):
    @mock.patch("seer.app.get_seer_project_preference")
    def test_get_seer_project_preference_endpoint(self, mock_get_preference):
        """Test the get_seer_project_preference_endpoint"""
        # Setup mock response
        repos = [
            RepoDefinition(
                owner="test-owner",
                name="test-repo",
                external_id="test-external-id",
                provider="github",
            )
        ]
        preference = SeerProjectPreference(
            organization_id=123,
            project_id=456,
            repositories=repos,
        )
        mock_get_preference.return_value = GetSeerProjectPreferenceResponse(preference=preference)

        # Make a request to the endpoint
        request_data = GetSeerProjectPreferenceRequest(project_id=456)
        response = app.test_client().post(
            "/v1/project-preference",
            data=request_data.json(),
            content_type="application/json",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.get_data(as_text=True))
        self.assertIsNotNone(response_data["preference"])
        self.assertEqual(response_data["preference"]["project_id"], 456)
        self.assertEqual(response_data["preference"]["organization_id"], 123)

        # Verify function was called with correct arguments
        mock_get_preference.assert_called_once()
        self.assertEqual(mock_get_preference.call_args[0][0].project_id, 456)

    @mock.patch("seer.app.set_seer_project_preference")
    def test_set_seer_project_preference_endpoint(self, mock_set_preference):
        """Test the set_seer_project_preference_endpoint"""
        # Setup mock response
        repos = [
            RepoDefinition(
                owner="test-owner",
                name="test-repo",
                external_id="test-external-id",
                provider="github",
                branch_name="feature-branch",
                instructions="# Test instructions",
            )
        ]
        preference = SeerProjectPreference(
            organization_id=123,
            project_id=456,
            repositories=repos,
        )
        mock_set_preference.return_value = SetSeerProjectPreferenceResponse(preference=preference)

        # Make a request to the endpoint
        request_data = SetSeerProjectPreferenceRequest(preference=preference)
        response = app.test_client().post(
            "/v1/project-preference/set",
            data=request_data.json(),
            content_type="application/json",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.get_data(as_text=True))
        self.assertIsNotNone(response_data["preference"])
        self.assertEqual(response_data["preference"]["project_id"], 456)
        self.assertEqual(response_data["preference"]["organization_id"], 123)
        self.assertEqual(
            response_data["preference"]["repositories"][0]["branch_name"], "feature-branch"
        )
        self.assertEqual(
            response_data["preference"]["repositories"][0]["instructions"], "# Test instructions"
        )

        # Verify function was called with correct arguments
        mock_set_preference.assert_called_once()
        self.assertEqual(mock_set_preference.call_args[0][0].preference.project_id, 456)
