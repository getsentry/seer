from unittest.mock import MagicMock, patch

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


class TestFileSystem:
    @patch("seer.automation.autofix.tools.cleanup_dir")
    def test_context_manager_cleanup(self, mock_cleanup_dir):
        context = MagicMock()

        with BaseTools(context) as tools:
            tools.tmp_dir = "/tmp/test_dir"
            tools.tmp_repo_dir = "/tmp/test_dir/repo"

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert tools.tmp_dir is None
        assert tools.tmp_repo_dir is None

    def test_cleanup_method(self):
        context = MagicMock()
        tools = BaseTools(context)
        tools.tmp_dir = "/tmp/test_dir"
        tools.tmp_repo_dir = "/tmp/test_dir/repo"

        with patch("seer.automation.autofix.tools.cleanup_dir") as mock_cleanup_dir:
            tools.cleanup()

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert tools.tmp_dir is None
        assert tools.tmp_repo_dir is None

    def test_cleanup_not_called_when_tmp_dir_is_none(self):
        context = MagicMock()
        tools = BaseTools(context)

        with patch("seer.automation.autofix.tools.cleanup_dir") as mock_cleanup_dir:
            tools.cleanup()

        mock_cleanup_dir.assert_not_called()
