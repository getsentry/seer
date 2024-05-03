import unittest
from unittest.mock import MagicMock, patch

from github import UnknownObjectException
from pydantic import ValidationError

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import InitializationError, RepoDefinition


class TestRepoClient(unittest.TestCase):
    @patch("seer.automation.codebase.repo_client.Github")
    @patch("seer.automation.codebase.repo_client.get_github_auth")
    def test_repo_client_accepts_github_provider(self, mock_get_github_auth, mock_github):
        # Mocking Github class and get_github_auth function to simulate GitHub API responses and authentication
        mock_github_instance = mock_github.return_value
        mock_github_instance.get_repo.return_value.default_branch = "main"
        mock_get_github_auth.return_value = (
            None  # Assuming get_github_auth returns None for simplicity
        )
        client = RepoClient(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertEqual(client.provider, "github")

    def test_repo_definition_rejects_unsupported_provider(self):
        with self.assertRaises(ValidationError):
            RepoDefinition(
                provider="unsupported_provider",
                owner="getsentry",
                name="seer",
                external_id="123",
            )

    @patch(
        "seer.automation.codebase.repo_client.get_app_installation",
        side_effect=UnknownObjectException(404, "Not Found"),
    )
    def test_repo_access_check_failed(self, mock_get_app_installation):
        result = RepoClient.check_repo_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)

    @patch(
        "seer.automation.codebase.repo_client.get_app_installation",
        return_value=(
            MagicMock(),
            MagicMock(
                raw_data={
                    "permissions": {
                        "contents": "write",
                        "metadata": "read",
                        "pull_requests": "write",
                    }
                }
            ),
        ),
    )
    def test_repo_access_check_success(self, mock_get_app_installation):
        result = RepoClient.check_repo_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertTrue(result)

    @patch(
        "seer.automation.codebase.repo_client.get_app_installation",
        return_value=(
            MagicMock(),
            MagicMock(
                raw_data={
                    "permissions": {
                        "contents": "read",
                        "metadata": "read",
                        "pull_requests": "write",
                    }
                }
            ),
        ),
    )
    def test_repo_access_check_insufficient_permissions(self, mock_get_app_installation):
        result = RepoClient.check_repo_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)

    @patch(
        "seer.automation.codebase.repo_client.get_app_installation",
        return_value=(
            MagicMock(),
            MagicMock(raw_data={}),
        ),
    )
    def test_repo_access_check_no_permissions(self, mock_get_app_installation):
        result = RepoClient.check_repo_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)
