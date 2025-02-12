from unittest.mock import MagicMock, patch

import pytest

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType


@pytest.fixture
def autofix_tools():
    context = MagicMock(AutofixContext)
    context.event_manager = MagicMock()
    return BaseTools(context)


class TestFileSearch:
    def test_file_search_found(self, autofix_tools: BaseTools):
        # Set up a dummy repo so that _get_repo_names() returns "testowner/testrepo"
        dummy_repo = MagicMock(owner="testowner", name="testrepo")
        dummy_repo.owner = "testowner"
        dummy_repo.name = "testrepo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/file2.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search("file2.py")
        # Expected formatting: a newline, header with repo name, then sorted matches (note that
        # "src/subfolder/file2.py" comes before "tests/file2.py" alphabetically).
        expected = (
            "\n FILES IN REPO testowner/testrepo:\n  src/subfolder/file2.py\n  tests/file2.py"
        )
        assert result == expected

    def test_file_search_not_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(owner="testowner", name="testrepo")
        dummy_repo.owner = "testowner"
        dummy_repo.name = "testrepo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/file2.py",
            "src/subfolder/file3.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search("nonexistent.py")
        expected = "no file with name nonexistent.py found in any repository"
        assert result == expected

    def test_file_search_with_repo_name(self, autofix_tools: BaseTools):
        # Instead of passing repo_name directly, set the context's repos.
        dummy_repo = MagicMock(owner="owner", name="test_repo")
        dummy_repo.owner = "owner"
        dummy_repo.name = "test_repo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {"src/file1.py"}
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.repo_client_type = RepoClientType.READ

        # Call file_search without a repo_name parameter.
        autofix_tools.file_search("file1.py")
        autofix_tools.context.get_repo_client.assert_any_call(
            repo_name="owner/test_repo", type=RepoClientType.READ
        )


class TestFileSearchWildcard:
    def test_file_search_wildcard_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(owner="testowner", name="testrepo")
        dummy_repo.owner = "testowner"
        dummy_repo.name = "testrepo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search_wildcard("*.py")
        expected = "\n FILES IN REPO testowner/testrepo:\n  src/file1.py\n  src/subfolder/file2.py\n  tests/test_file1.py"
        assert result == expected

    def test_file_search_wildcard_not_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(owner="testowner", name="testrepo")
        dummy_repo.owner = "testowner"
        dummy_repo.name = "testrepo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search_wildcard("*.js")
        expected = "No files matching pattern '*.js' found in any repository"
        assert result == expected

    def test_file_search_wildcard_with_repo_name(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(owner="owner", name="test_repo")
        dummy_repo.owner = "owner"
        dummy_repo.name = "test_repo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {"src/file1.py"}
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.repo_client_type = RepoClientType.READ

        autofix_tools.file_search_wildcard("*.py")
        autofix_tools.context.get_repo_client.assert_any_call(
            repo_name="owner/test_repo", type=RepoClientType.READ
        )


class TestFileSystem:
    @patch("seer.automation.autofix.tools.cleanup_dir")
    def test_context_manager_cleanup(self, mock_cleanup_dir):
        context = MagicMock()

        with BaseTools(context) as tools:
            # Set tmp_dir as a dictionary mapping a dummy key to the tuple of paths.
            tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}
            tools.tmp_repo_dir = "/tmp/test_dir/repo"

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert tools.tmp_dir is None
        # Since cleanup() does not clear tmp_repo_dir, we expect it to remain unchanged.
        assert tools.tmp_repo_dir == "/tmp/test_dir/repo"

    def test_cleanup_method(self):
        context = MagicMock()
        tools = BaseTools(context)
        tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        tools.tmp_repo_dir = "/tmp/test_dir/repo"

        with patch("seer.automation.autofix.tools.cleanup_dir") as mock_cleanup_dir:
            tools.cleanup()

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert tools.tmp_dir is None
        # tmp_repo_dir is not cleared by cleanup(), so it should remain unchanged.
        assert tools.tmp_repo_dir == "/tmp/test_dir/repo"

    def test_cleanup_not_called_when_tmp_dir_is_none(self):
        context = MagicMock()
        tools = BaseTools(context)

        with patch("seer.automation.autofix.tools.cleanup_dir") as mock_cleanup_dir:
            tools.cleanup()

        mock_cleanup_dir.assert_not_called()


class TestSemanticFileSearch:
    def test_semantic_file_search_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(owner="owner", name="test_repo")
        dummy_repo.owner = "owner"
        dummy_repo.name = "test_repo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        # Files available in the repo
        mock_repo_client.get_valid_file_paths.return_value = [
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        ]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.get_file_contents.return_value = "test file contents"

        mock_llm_client = MagicMock()
        # The parsed response now includes both file_path and the repo_name built from the dummy repo.
        mock_llm_client.generate_structured.return_value.parsed = MagicMock(
            file_path="src/file1.py", repo_name="owner/test_repo"
        )

        result = autofix_tools.semantic_file_search(
            "find the main file", llm_client=mock_llm_client
        )
        expected = "This file might be what you're looking for: `src/file1.py`. Contents:\n\ntest file contents"
        assert result == expected

    def test_semantic_file_search_not_found_no_file_path(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(owner="owner", name="test_repo")
        dummy_repo.owner = "owner"
        dummy_repo.name = "test_repo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_valid_file_paths.return_value = [
            "src/file1.py",
            "tests/test_file1.py",
        ]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value.parsed = None

        result = autofix_tools.semantic_file_search(
            "find nonexistent file", llm_client=mock_llm_client
        )
        expected = "Could not figure out which file matches what you were looking for. You'll have to try yourself."
        assert result == expected

    def test_semantic_file_search_not_found_no_contents(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(owner="owner", name="test_repo")
        dummy_repo.owner = "owner"
        dummy_repo.name = "test_repo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_valid_file_paths.return_value = [
            "src/file1.py",
            "tests/test_file1.py",
        ]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.get_file_contents.return_value = None

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value.parsed = MagicMock(
            file_path="src/file1.py", repo_name="owner/test_repo"
        )

        result = autofix_tools.semantic_file_search(
            "find file with no contents", llm_client=mock_llm_client
        )
        expected = "Could not figure out which file matches what you were looking for. You'll have to try yourself."
        assert result == expected

    def test_semantic_file_search_with_repo_name(self, autofix_tools: BaseTools):
        # Instead of passing repo_name directly as an argument,
        # set context.repos so that _get_repo_names() returns "owner/specific_repo"
        dummy_repo = MagicMock(owner="owner", name="specific_repo")
        dummy_repo.owner = "owner"
        dummy_repo.name = "specific_repo"
        autofix_tools.context.repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_valid_file_paths.return_value = ["src/file1.py"]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.get_file_contents.return_value = "test file contents"
        autofix_tools.repo_client_type = RepoClientType.READ

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value.parsed = MagicMock(
            file_path="src/file1.py", repo_name="owner/specific_repo"
        )

        autofix_tools.semantic_file_search("find file", llm_client=mock_llm_client)
        autofix_tools.context.get_repo_client.assert_any_call(
            repo_name="owner/specific_repo", type=RepoClientType.READ
        )
