import unittest
from unittest.mock import patch

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import InitializationError


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
        client = RepoClient(repo_provider="github", repo_owner="test_owner", repo_name="test_repo")
        self.assertEqual(client.provider, "github")

    @patch("seer.automation.codebase.repo_client.Github")
    @patch("seer.automation.codebase.repo_client.get_github_auth")
    def test_repo_client_accepts_integrations_github_provider(
        self, mock_get_github_auth, mock_github
    ):
        # Mocking Github class and get_github_auth function to simulate GitHub API responses and authentication
        mock_github_instance = mock_github.return_value
        mock_github_instance.get_repo.return_value.default_branch = "main"
        mock_get_github_auth.return_value = (
            None  # Assuming get_github_auth returns None for simplicity
        )
        client = RepoClient(
            repo_provider="integrations:github", repo_owner="test_owner", repo_name="test_repo"
        )
        self.assertEqual(client.provider, "github")

    @patch("seer.automation.codebase.repo_client.get_github_auth")
    def test_repo_client_rejects_unsupported_provider(self, mock_get_github_auth):
        mock_get_github_auth.return_value = (
            None  # Assuming get_github_auth returns None for simplicity
        )
        with self.assertRaises(InitializationError):
            RepoClient(
                repo_provider="unsupported_provider", repo_owner="test_owner", repo_name="test_repo"
            )
