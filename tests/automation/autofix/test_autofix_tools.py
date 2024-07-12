import textwrap
from unittest.mock import MagicMock, patch

import pytest

from seer.automation.autofix.components.snippet_replacement import SnippetReplacementOutput
from seer.automation.autofix.tools import BaseTools, CodeActionTools
from seer.automation.models import FileChange


class TestReplaceSnippetWith:
    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.fixture
    def code_action_tools(self, mock_context):
        return CodeActionTools(context=mock_context)

    @patch(
        "seer.automation.autofix.components.snippet_replacement.SnippetReplacementComponent.invoke"
    )
    def test_replace_snippet_with_success(self, mock_invoke, mock_context, code_action_tools):
        original_snippet = "print('Hello, world!')"
        replacement_snippet = "print('Goodbye, world!')"
        file_path = "test_file.py"
        contents = textwrap.dedent(
            """\
            def foo():
                print('Hello, world!')
                return True
            """
        )
        mock_context.get_file_contents.return_value = contents

        code = textwrap.dedent(
            """\
            def foo():
                print('Goodbye, world!')
                return True
            """
        )
        mock_invoke.return_value = SnippetReplacementOutput(snippet=code)

        code_action_tools.store_file_change = MagicMock()

        result = code_action_tools.replace_snippet_with(
            file_path, "repo", original_snippet, replacement_snippet, "message"
        )

        assert result == f"success: Resulting code after replacement:\n```\n{code}\n```\n"
        code_action_tools.store_file_change.assert_called_once_with(
            "repo",
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=contents,
                new_snippet=code,
                description="message",
            ),
        )

    @patch(
        "seer.automation.autofix.components.snippet_replacement.SnippetReplacementComponent.invoke"
    )
    def test_replace_snippet_with_chunk_newlines(
        self, mock_invoke, mock_context, code_action_tools
    ):
        original_snippet = "print('Hello, world!')"
        replacement_snippet = "print('Goodbye, world!')"
        file_path = "test_file.py"
        contents = textwrap.dedent(
            """\

            def foo():
                print('Hello, world!')
                return True


            """
        )
        mock_context.get_file_contents.return_value = contents

        code = textwrap.dedent(
            """\
            def foo():
                print('Goodbye, world!')
                return True"""
        )
        mock_invoke.return_value = SnippetReplacementOutput(snippet=code)

        code_action_tools.store_file_change = MagicMock()

        result = code_action_tools.replace_snippet_with(
            file_path, "repo", original_snippet, replacement_snippet, "message"
        )

        assert result == f"success: Resulting code after replacement:\n```\n{code}\n```\n"
        code_action_tools.store_file_change.assert_called_once_with(
            "repo",
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=contents.strip("\n") + "\n",
                new_snippet=code + "\n",
                description="message",
            ),
        )

        expected_final_document_content = textwrap.dedent(
            """\

            def foo():
                print('Goodbye, world!')
                return True


            """
        )
        assert (
            code_action_tools.store_file_change.call_args[0][1].apply(contents)
            == expected_final_document_content
        )


class TestListDirectory:
    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.fixture
    def base_tools(self, mock_context):
        return BaseTools(context=mock_context)

    def test_list_directory_empty(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = set()
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("/some/path")
        assert result == "<no entries found in directory '/some/path'>"

    def test_list_directory_root(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "file1.txt",
            "file2.py",
            "dir1/",
            "dir2/subdir/file3.js",
        }
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("/")
        expected = textwrap.dedent(
            """\
        <entries>
        Directories:
          dir1/
          dir2/

        Files:
          file1.txt
          file2.py
        </entries>"""
        )
        assert result == expected

    def test_list_directory_subdirectory(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "dir1/file1.txt",
            "dir1/subdir/file2.py",
            "dir1/subdir/subsubdir/file3.js",
        }
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("dir1")
        expected = textwrap.dedent(
            """\
        <entries>
        Directories:
          subdir/

        Files:
          file1.txt
        </entries>"""
        )
        assert result == expected

    def test_list_directory_only_files(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "dir1/file1.txt",
            "dir1/file2.py",
            "dir1/file3.js",
        }
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("dir1")
        expected = textwrap.dedent(
            """\
        <entries>
        Files:
          file1.txt
          file2.py
          file3.js
        </entries>"""
        )
        assert result == expected

    def test_list_directory_only_directories(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "dir1/subdir1/",
            "dir1/subdir2/",
            "dir1/subdir3/",
        }
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("dir1")
        expected = textwrap.dedent(
            """\
        <entries>
        Directories:
          subdir1/
          subdir2/
          subdir3/
        </entries>"""
        )
        assert result == expected

    def test_list_directory_with_trailing_slash(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "dir1/file1.txt",
            "dir1/subdir/file2.py",
        }
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("dir1/")
        expected = textwrap.dedent(
            """\
        <entries>
        Directories:
          subdir/

        Files:
          file1.txt
        </entries>"""
        )
        assert result == expected

    def test_list_directory_nonexistent_path(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "dir1/file1.txt",
            "dir2/file2.py",
        }
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("nonexistent")
        assert result == "<no entries found in directory 'nonexistent'>"

    def test_list_directory_with_repo_name(self, mock_context, base_tools):
        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "file1.txt",
            "dir1/file2.py",
        }
        mock_context.get_repo_client.return_value = mock_repo_client

        result = base_tools.list_directory("/", repo_name="test_repo")
        expected = textwrap.dedent(
            """\
        <entries>
        Directories:
          dir1/

        Files:
          file1.txt
        </entries>"""
        )
        assert result == expected
        mock_context.get_repo_client.assert_called_once_with(repo_name="test_repo")
