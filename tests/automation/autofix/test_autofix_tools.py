from unittest.mock import MagicMock

import pytest

from seer.automation.autofix.tools import BaseTools


@pytest.fixture
def autofix_tools():
    context = MagicMock()
    return BaseTools(context)


class TestFileSearch:
    def test_file_search_found(self, autofix_tools: BaseTools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/file2.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search("file2.py")
        assert result == "src/subfolder/file2.py,tests/file2.py"

    def test_file_search_not_found(self, autofix_tools: BaseTools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/file2.py",
            "src/subfolder/file3.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search("nonexistent.py")
        assert result == "no file with name nonexistent.py found in repository"

    def test_file_search_with_repo_name(self, autofix_tools: BaseTools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {"src/file1.py"}
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        autofix_tools.file_search("file1.py", repo_name="test_repo")
        autofix_tools.context.get_repo_client.assert_called_once_with(repo_name="test_repo")


class TestFileSearchWildcard:
    def test_file_search_wildcard_found(self, autofix_tools: BaseTools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search_wildcard("*.py")
        assert result == "src/file1.py\nsrc/subfolder/file2.py\ntests/test_file1.py"

    def test_file_search_wildcard_not_found(self, autofix_tools: BaseTools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search_wildcard("*.js")
        assert result == "No files matching pattern '*.js' found in repository"

    def test_file_search_wildcard_with_repo_name(self, autofix_tools: BaseTools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {"src/file1.py"}
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        autofix_tools.file_search_wildcard("*.py", repo_name="test_repo")
        autofix_tools.context.get_repo_client.assert_called_once_with(repo_name="test_repo")
