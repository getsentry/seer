import os
import unittest
from unittest.mock import MagicMock, patch

from github import UnknownObjectException
from pydantic import ValidationError

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import RepoDefinition


class TestRepoClient(unittest.TestCase):
    @patch("seer.automation.codebase.repo_client.Github")
    @patch("seer.automation.codebase.repo_client.get_github_app_auth_and_installation")
    def test_repo_client_accepts_github_provider(self, mock_get_github_auth, mock_github):
        os.environ["GITHUB_APP_ID"] = "1337"
        # Mocking Github class and get_github_auth function to simulate GitHub API responses and authentication
        mock_github_instance = mock_github.return_value
        mock_github_instance.get_repo.return_value.default_branch = "main"
        client = RepoClient.from_repo_definition(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "read",
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
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        side_effect=UnknownObjectException(404, "Not Found"),
    )
    def test_write_repo_access_check_failed(self, mock_get_app_installation):
        result = RepoClient.check_repo_write_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
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
    def test_write_repo_access_check_success(self, mock_get_app_installation):
        result = RepoClient.check_repo_write_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertTrue(result)

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
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
    def test_write_repo_access_check_insufficient_permissions(self, mock_get_app_installation):
        result = RepoClient.check_repo_write_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=(
            MagicMock(),
            MagicMock(raw_data={}),
        ),
    )
    def test_write_repo_access_check_no_permissions(self, mock_get_app_installation):
        result = RepoClient.check_repo_write_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        side_effect=UnknownObjectException(404, "Not Found"),
    )
    def test_read_repo_access_check_failed(self, mock_get_app_installation):
        result = RepoClient.check_repo_read_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=(
            MagicMock(),
            MagicMock(
                raw_data={
                    "permissions": {
                        "contents": "write",
                    }
                }
            ),
        ),
    )
    def test_read_repo_access_check_success(self, mock_get_app_installation):
        result = RepoClient.check_repo_read_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertTrue(result)

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=(
            MagicMock(),
            MagicMock(
                raw_data={
                    "permissions": {
                        "pull_requests": "write",
                    }
                }
            ),
        ),
    )
    def test_read_repo_access_check_insufficient_permissions(self, mock_get_app_installation):
        result = RepoClient.check_repo_read_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=(
            MagicMock(),
            MagicMock(raw_data={}),
        ),
    )
    def test_read_repo_access_check_no_permissions(self, mock_get_app_installation):
        result = RepoClient.check_repo_read_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        self.assertFalse(result)


class TestRepoClientIndexFileSet(unittest.TestCase):
    @patch("seer.automation.codebase.repo_client.Github")
    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=[MagicMock(), MagicMock()],
    )
    def test_all_files_included(self, get_github_app_auth_and_installation, mock_Github):
        mock_tree = MagicMock(
            tree=[
                MagicMock(path="file1.py", mode="100644", type="blob", size=1 * 1024 * 1024),
                MagicMock(path="file2.py", mode="100644", type="blob", size=1 * 1024 * 1024),
            ],
            raw_data={"truncated": False},
        )

        mock_github_instance = mock_Github.return_value
        mock_github_instance.get_repo.return_value.get_git_tree.return_value = mock_tree
        client = RepoClient(
            1,
            "very private heh",
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
        )
        result = client.get_index_file_set("main")
        assert result == {"file1.py", "file2.py"}

    @patch("seer.automation.codebase.repo_client.Github")
    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=[MagicMock(), MagicMock()],
    )
    def test_filters_out_folders(self, mock_get_github_auth, mock_Github):
        mock_tree = MagicMock(
            tree=[
                MagicMock(path="file1.py", mode="100644", type="blob", size=1 * 1024 * 1024),
                MagicMock(path="folder", mode="100644", type="tree", size=1 * 1024 * 1024),
            ],
            raw_data={"truncated": False},
        )

        mock_github_instance = mock_Github.return_value
        mock_github_instance.get_repo.return_value.get_git_tree.return_value = mock_tree
        client = RepoClient(
            1,
            "very private heh",
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
        )
        result = client.get_index_file_set("main")
        assert result == {"file1.py"}

    @patch("seer.automation.codebase.repo_client.Github")
    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=[MagicMock(), MagicMock()],
    )
    def test_filters_out_symlinks(self, mock_get_github_auth, mock_Github):
        mock_tree = MagicMock(
            tree=[
                MagicMock(path="file1.py", mode="100644", type="blob", size=1 * 1024 * 1024),
                MagicMock(path="symlink", mode="120000", type="blob", size=1 * 1024 * 1024),
            ],
            raw_data={"truncated": False},
        )

        mock_github_instance = mock_Github.return_value
        mock_github_instance.get_repo.return_value.get_git_tree.return_value = mock_tree
        client = RepoClient(
            1,
            "very private heh",
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
        )
        result = client.get_index_file_set("main")
        assert result == {"file1.py"}

    @patch("seer.automation.codebase.repo_client.Github")
    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=[MagicMock(), MagicMock()],
    )
    def test_filters_out_unknown_file_types(self, mock_get_github_auth, mock_Github):
        mock_tree = MagicMock(
            tree=[
                MagicMock(path="file1.py", mode="100644", type="blob", size=1 * 1024 * 1024),
                MagicMock(path="file2.hjk", mode="100644", type="blob", size=1 * 1024 * 1024),
            ],
            raw_data={"truncated": False},
        )

        mock_github_instance = mock_Github.return_value
        mock_github_instance.get_repo.return_value.get_git_tree.return_value = mock_tree
        client = RepoClient(
            1,
            "very private heh",
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
        )
        result = client.get_index_file_set("main")
        assert result == {"file1.py"}

    @patch("seer.automation.codebase.repo_client.Github")
    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        return_value=[MagicMock(), MagicMock()],
    )
    def test_filters_out_large_files(self, mock_get_github_auth, mock_Github):
        mock_tree = MagicMock(
            tree=[
                MagicMock(path="file1.py", mode="100644", type="blob", size=1 * 1024 * 1024),
                MagicMock(path="file2.py", mode="100644", type="blob", size=4 * 1024 * 1024),
            ],
            raw_data={"truncated": False},
        )

        mock_github_instance = mock_Github.return_value
        mock_github_instance.get_repo.return_value.get_git_tree.return_value = mock_tree
        client = RepoClient(
            1,
            "very private heh",
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
        )
        result = client.get_index_file_set("main")
        assert result == {"file1.py"}
