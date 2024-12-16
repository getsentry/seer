from unittest.mock import ANY, MagicMock, patch

import pytest
from github import UnknownObjectException
from pydantic import ValidationError

from seer.automation.codebase.models import GithubPrReviewComment
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import RepoDefinition


@pytest.fixture(autouse=True)
def clear_repo_client_cache():
    """Clear the RepoClient.from_repo_definition cache before each test"""
    RepoClient.from_repo_definition.cache_clear()
    yield


@pytest.fixture
def mock_github():
    with patch("seer.automation.codebase.repo_client.Github") as mock:
        mock_instance = mock.return_value
        mock_instance.get_repo.return_value.default_branch = "main"
        mock_instance.get_repo.return_value.get_branch.return_value.commit.sha = "default_sha"
        yield mock_instance


@pytest.fixture
def mock_get_github_auth():
    with patch("seer.automation.codebase.repo_client.get_github_app_auth_and_installation") as mock:
        yield mock


@pytest.fixture
def repo_definition():
    return RepoDefinition(
        provider="github",
        owner="getsentry",
        name="seer",
        external_id="123",
        base_commit_sha="test_sha",
    )


@pytest.fixture
def repo_client(mock_github, mock_get_github_auth, repo_definition):
    return RepoClient.from_repo_definition(repo_definition, "read")


class TestRepoClient:

    def test_repo_client_initialization(self, repo_client):
        assert repo_client.provider == "github"
        assert repo_client.repo_owner == "getsentry"
        assert repo_client.repo_name == "seer"
        assert repo_client.repo_external_id == "123"
        assert repo_client.base_commit_sha == "test_sha"

    def test_repo_client_initialization_without_base_commit_sha(
        self, mock_github, mock_get_github_auth
    ):
        repo_def_without_sha = RepoDefinition(
            provider="github", owner="getsentry", name="seer", external_id="123"
        )
        client = RepoClient.from_repo_definition(repo_def_without_sha, "read")
        assert client.base_commit_sha == "default_sha"

    def test_repo_client_accepts_github_provider(self, mock_github, mock_get_github_auth):
        client = RepoClient.from_repo_definition(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "read",
        )
        assert client.provider == "github"

    def test_repo_definition_rejects_unsupported_provider(self):
        with pytest.raises(ValidationError):
            RepoDefinition(
                provider="unsupported_provider",
                owner="getsentry",
                name="seer",
                external_id="123",
            )

    @patch("seer.automation.codebase.repo_client.requests.get")
    @patch("seer.automation.codebase.repo_client.tarfile.open")
    def test_load_repo_to_tmp_dir(self, mock_tarfile, mock_requests, repo_client, tmp_path):
        mock_requests.return_value.status_code = 200
        mock_requests.return_value.content = b"test_content"

        with patch(
            "seer.automation.codebase.repo_client.tempfile.mkdtemp", return_value=str(tmp_path)
        ):
            tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()

        assert tmp_dir == str(tmp_path)
        assert tmp_repo_dir == str(tmp_path / "repo")
        mock_requests.assert_called_once()
        mock_tarfile.assert_called_once()

    @patch("seer.automation.codebase.repo_client.requests.get")
    def test_get_file_content(self, mock_requests, repo_client, mock_github):
        mock_content = MagicMock()
        mock_content.decoded_content = b"test content"
        mock_github.get_repo.return_value.get_contents.return_value = mock_content

        content, _ = repo_client.get_file_content("test_file.py")

        assert content == "test content"
        mock_github.get_repo.return_value.get_contents.assert_called_with(
            "test_file.py", ref="test_sha"
        )

    @patch("seer.automation.codebase.repo_client.requests.get")
    def test_fail_get_file_content(self, mock_requests, repo_client, mock_github):
        mock_content = MagicMock()
        mock_content.decoded_content = b"test content"
        # this is a list of contents, so the content returned should be None
        mock_github.get_repo.return_value.get_contents\
            .return_value = [mock_content, mock_content]

        content, encoding = repo_client.get_file_content("test_file.py")

        assert not content
        assert encoding == "utf-8"
        mock_github.get_repo.return_value.get_contents.assert_called_with(
            "test_file.py", ref="test_sha"
        )

    def test_get_valid_file_paths(self, repo_client, mock_github):
        mock_tree = MagicMock()
        mock_tree.tree = [MagicMock(path="file1.py"), MagicMock(path="file2.py")]
        mock_tree.raw_data = {"truncated": False}
        mock_github.get_repo.return_value.get_git_tree.return_value = mock_tree

        file_paths = repo_client.get_valid_file_paths()

        assert file_paths == {"file1.py", "file2.py"}
        mock_github.get_repo.return_value.get_git_tree.assert_called_with(
            "test_sha", recursive=True
        )

    def test_get_index_file_set(self, repo_client, mock_github):
        mock_tree = MagicMock()
        mock_tree.tree = [
            MagicMock(path="file1.py", type="blob", size=1000, mode="100644"),
            MagicMock(path="file2.py", type="blob", size=1000, mode="100644"),
            MagicMock(path="large_file.py", type="blob", size=3 * 1024 * 1024, mode="100644"),
            MagicMock(path="dir", type="tree"),
        ]
        mock_tree.raw_data = {"truncated": False}
        mock_github.get_repo.return_value.get_git_tree.return_value = mock_tree

        file_set = repo_client.get_index_file_set()

        assert file_set == {"file1.py", "file2.py"}
        mock_github.get_repo.return_value.get_git_tree.assert_called_with(
            sha="test_sha", recursive=True
        )

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        side_effect=UnknownObjectException(404, "Not Found"),
    )
    def test_write_repo_access_check_failed(self, mock_get_app_installation):
        result = RepoClient.check_repo_write_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        assert result is False

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
        assert result is True

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
        assert result is False

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
        assert result is False

    @patch(
        "seer.automation.codebase.repo_client.get_github_app_auth_and_installation",
        side_effect=UnknownObjectException(404, "Not Found"),
    )
    def test_read_repo_access_check_failed(self, mock_get_app_installation):
        result = RepoClient.check_repo_read_access(
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123")
        )
        assert result is False

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
        assert result is True

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
        assert result is False

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
        assert result is False

    @patch("seer.automation.codebase.repo_client.requests.get")
    def test_get_pr_diff_content(self, mock_requests, repo_client):
        mock_requests.return_value.text = "Mock diff content"
        mock_requests.return_value.raise_for_status = MagicMock()

        diff_content = repo_client.get_pr_diff_content(
            "https://api.github.com/repos/owner/repo/pulls/1"
        )

        assert diff_content == "Mock diff content"
        mock_requests.assert_called_once()
        mock_requests.return_value.raise_for_status.assert_called_once()

    @patch("seer.automation.codebase.repo_client.requests.post")
    def test_comment_root_cause_on_pr_for_copilot(self, mock_requests, repo_client):
        mock_requests.return_value.raise_for_status = MagicMock()

        repo_client.comment_root_cause_on_pr_for_copilot(
            "https://github.com/owner/repo/pull/1", run_id=123, issue_id=456, comment="Test comment"
        )

        mock_requests.assert_called_once()
        mock_requests.return_value.raise_for_status.assert_called_once()

        # Check if the correct URL and data were used in the request
        args, kwargs = mock_requests.call_args
        assert args[0] == "https://api.github.com/repos/owner/repo/issues/1/comments"
        assert kwargs["json"]["body"] == "Test comment"
        assert (
            kwargs["json"]["actions"][0]["prompt"]
            == "@sentry find a fix for issue 456 with run ID 123"
        )

    @patch("seer.automation.codebase.repo_client.requests.post")
    def test_comment_pr_generated_for_copilot(self, mock_requests, repo_client):
        mock_requests.return_value.raise_for_status = MagicMock()

        repo_client.comment_pr_generated_for_copilot(
            "https://github.com/owner/repo/pull/1",
            "https://github.com/owner/repo/pull/2",
            run_id=123,
        )

        mock_requests.assert_called_once()
        mock_requests.return_value.raise_for_status.assert_called_once()

        # Check if the correct URL and data were used in the request
        args, kwargs = mock_requests.call_args
        assert args[0] == "https://api.github.com/repos/owner/repo/issues/1/comments"
        assert (
            "A fix has been generated and is available [here](https://github.com/owner/repo/pull/2) for your review."
            in kwargs["json"]["body"]
        )
        assert "Autofix Run ID: 123" in kwargs["json"]["body"]

    @patch("seer.automation.codebase.repo_client.requests.get")
    def test_get_pr_head_sha(self, mock_requests, repo_client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"head": {"sha": "abcdef1234567890"}}
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        head_sha = repo_client.get_pr_head_sha("https://api.github.com/repos/owner/repo/pulls/1")

        assert head_sha == "abcdef1234567890"
        mock_requests.assert_called_once()
        mock_response.raise_for_status.assert_called_once()

    @patch("seer.automation.codebase.repo_client.requests.get")
    def test_get_pr_head_sha_error(self, mock_requests, repo_client):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_requests.return_value = mock_response

        with pytest.raises(Exception, match="API error"):
            repo_client.get_pr_head_sha("https://api.github.com/repos/owner/repo/pulls/1")

        mock_requests.assert_called_once()
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.parametrize(
        "patch_type,path,expected_sha",
        [
            ("create", "new_file.py", "new_blob_sha"),
            ("edit", "existing.py", "modified_blob_sha"),
            ("delete", "delete_me.py", None),
            ("edit", "/leading/slash.py", "modified_blob_sha"),
        ],
    )
    def test_get_one_file_autofix_change_combinations(
        self, repo_client, mock_github, patch_type, path, expected_sha
    ):
        # mock the blob creation
        mock_blob = MagicMock(sha=expected_sha)
        mock_github.get_repo.return_value.create_git_blob.return_value = mock_blob
        mock_content_file = MagicMock(decoded_content=b"content")
        repo_client.repo.get_contents.return_value = (
            mock_content_file if patch_type != "create" else None
        )

        patch = MagicMock(
            **{
                "path": path,
                "type": patch_type,
                "apply.return_value": "new content" if expected_sha else None,
            }
        )

        # Execute
        result = repo_client.process_one_file_for_git_commit(branch_ref="main", patch=patch)

        expected_path = path[1:] if path.startswith("/") else path

        assert result._identity["path"] == expected_path
        assert result._identity["mode"] == "100644"
        assert result._identity["type"] == "blob"
        assert result._identity["sha"] == expected_sha

    def test_create_branch_from_changes_invalid_input(self, repo_client):
        # Test with no changes provided
        with pytest.raises(
            ValueError, match="Either file_patches or file_changes must be provided"
        ):
            repo_client.create_branch_from_changes(
                pr_title="Test PR", file_patches=None, file_changes=None
            )

    @pytest.mark.parametrize(
        "input_type,input_data",
        [
            (
                "patches",
                [
                    MagicMock(
                        **{"path": "test.py", "type": "edit", "apply.return_value": "new content"}
                    )
                ],
            ),
            (
                "changes",
                [
                    MagicMock(
                        **{
                            "path": "test.py",
                            "content": "new content",
                            "mode": "100644",
                            "type": "blob",
                        }
                    )
                ],
            ),
        ],
    )
    def test_create_branch_from_changes_success(self, repo_client, input_type, input_data):
        mock_comparison = MagicMock()
        mock_comparison.ahead_by = 1

        mock_branch_ref = MagicMock()
        mock_branch_ref.ref = "refs/heads/test-branch"
        mock_branch_ref.object.sha = "new-commit-sha"

        mock_blob = MagicMock(sha="new-commit-sha")

        repo_client.repo.create_git_blob.return_value = mock_blob
        repo_client.repo.compare.return_value = mock_comparison
        repo_client.repo._create_branch.return_value = mock_branch_ref
        repo_client.repo.get_default_branch_head_sha.return_value = "default-sha"

        # Test the method
        repo_client.create_branch_from_changes(
            pr_title="Test PR",
            file_patches=input_data if input_type == "patches" else None,
            file_changes=input_data if input_type == "changes" else None,
        )

        # Assertions
        repo_client.repo.create_git_tree.assert_called_once()
        repo_client.repo.create_git_commit.assert_called_once()

    def test_create_branch_from_changes_no_changes(self, repo_client):
        mock_comparison = MagicMock()
        # this is the case where the branch is up to date with the default branch
        mock_comparison.ahead_by = 0

        mock_branch_ref = MagicMock()
        mock_branch_ref.ref = "refs/heads/test-branch"
        mock_branch_ref.object.sha = "new-commit-sha"

        # mock the blob creation
        mock_blob = MagicMock(sha="blob-sha")

        repo_client.repo.create_git_blob.return_value = mock_blob
        repo_client.repo.compare.return_value = mock_comparison
        repo_client.repo._create_branch.return_value = mock_branch_ref
        repo_client.repo.get_default_branch_head_sha.return_value = "default-sha"

        # Test the method
        result = repo_client.create_branch_from_changes(
            pr_title="Test PR",
            file_patches=[
                MagicMock(
                    **{"path": "test.py", "type": "edit", "apply.return_value": "new content"}
                )
            ],
        )

        # Assertions
        assert not result  # branch was deleted
        repo_client.repo.create_git_tree.assert_called_once()
        repo_client.repo.create_git_commit.assert_called_once()

    @pytest.mark.parametrize(
        "patch", [(MagicMock(path="test.py", type=None)), (MagicMock(path=None, type="edit"))]
    )
    def test_get_one_file_autofix_change_invalid_input(self, repo_client, patch):
        with pytest.raises(ValueError):
            repo_client.process_one_file_for_git_commit(branch_ref="main", patch=patch)


class TestRepoClientIndexFileSet:
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

    @patch("seer.automation.codebase.repo_client.requests.post")
    def test_post_unit_test_reference_to_original_pr(self, mock_post, repo_client):
        original_pr_url = "https://github.com/sentry/sentry/pull/12345"
        unit_test_pr_url = "https://github.com/sentry/sentry/pull/67890"
        expected_url = "https://api.github.com/repos/sentry/sentry/issues/12345/comments"
        expected_comment = (
            f"Sentry has generated a new [PR]({unit_test_pr_url}) with unit tests for this PR. "
            f"View the new PR({unit_test_pr_url}) to review the changes."
        )
        expected_params = {"body": expected_comment}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "html_url": "https://github.com/sentry/sentry/pull/12345#issuecomment-1"
        }
        mock_post.return_value = mock_response

        result = repo_client.post_unit_test_reference_to_original_pr(
            original_pr_url, unit_test_pr_url
        )

        mock_post.assert_called_once_with(expected_url, headers=ANY, json=expected_params)

        assert result == "https://github.com/sentry/sentry/pull/12345#issuecomment-1"

    @patch("seer.automation.codebase.repo_client.requests.post")
    def test_post_unit_test_not_generated_message_to_original_pr(self, mock_post, repo_client):
        original_pr_url = "https://github.com/sentry/sentry/pull/12345"
        expected_url = "https://api.github.com/repos/sentry/sentry/issues/12345/comments"
        expected_comment = "Sentry has determined that unit tests already exist on this PR or that they are not necessary."
        expected_params = {"body": expected_comment}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "html_url": "https://github.com/sentry/sentry/pull/12345#issuecomment-1"
        }
        mock_post.return_value = mock_response

        result = repo_client.post_unit_test_not_generated_message_to_original_pr(original_pr_url)

        mock_post.assert_called_once_with(expected_url, headers=ANY, json=expected_params)

        assert result == "https://github.com/sentry/sentry/pull/12345#issuecomment-1"

    @patch("seer.automation.codebase.repo_client.requests.post")
    def test_post_issue_comment(self, mock_post, repo_client):
        pr_url = "https://github.com/repos/sentry/sentry/pulls/12345"
        expected_url = "https://api.github.com/repos/sentry/sentry/issues/12345/comments"
        expected_comment = "No changes requiring review at this time."
        expected_params = {"body": expected_comment}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "html_url": "https://github.com/sentry/sentry/pull/12345#issuecomment-1"
        }
        mock_post.return_value = mock_response

        result = repo_client.post_issue_comment(pr_url, expected_comment)

        mock_post.assert_called_once_with(expected_url, headers=ANY, json=expected_params)

        assert result == "https://github.com/sentry/sentry/pull/12345#issuecomment-1"

    @patch("seer.automation.codebase.repo_client.requests.post")
    def test_post_pr_review_comment(self, mock_post, repo_client):
        pr_url = "https://github.com/repos/sentry/sentry/pulls/12345"
        expected_url = "https://api.github.com/repos/sentry/sentry/pulls/12345/comments"
        comment = GithubPrReviewComment(path="file.py", line=10, body="Please fix this", start_line=None)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "html_url": "https://github.com/sentry/sentry/pull/12345#issuecomment-1"
        }
        mock_post.return_value = mock_response

        result = repo_client.post_pr_review_comment(pr_url, comment)

        mock_post.assert_called_once_with(expected_url, headers=ANY, json=comment)

        assert result == "https://github.com/sentry/sentry/pull/12345#issuecomment-1"
