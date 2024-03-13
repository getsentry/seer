import unittest
from unittest.mock import MagicMock

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.models import (
    EventDetails,
    ExceptionDetails,
    RepoDefinition,
    Stacktrace,
    StacktraceFrame,
)
from seer.automation.codebase.codebase_index import CodebaseIndex


class TestAutofixContext(unittest.TestCase):
    def setUp(self):
        self.mock_codebase_index = MagicMock()
        self.mock_repo_client = MagicMock()
        self.mock_codebase_index.repo_client = self.mock_repo_client
        self.autofix_context = AutofixContext(
            1,
            1,
            [],
            MagicMock(),
            MagicMock(),
        )
        self.autofix_context.get_codebase = MagicMock(return_value=self.mock_codebase_index)
        self.autofix_context.has_codebase_index = MagicMock(return_value=True)

    def test_diff_contains_stacktrace_files_with_intersection(self):
        # Mock the get_commit_file_diffs method to return changed and removed files
        self.mock_repo_client.get_commit_file_diffs.return_value = (
            ["file1.py", "file2.py"],
            ["file3.py"],
        )
        # Create a stacktrace with one of the files that has changed
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file2.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file2.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files
        self.assertTrue(self.autofix_context.diff_contains_stacktrace_files(1, event_details))

    def test_diff_contains_stacktrace_files_without_intersection(self):
        # Mock the get_commit_file_diffs method to return changed and removed files
        self.mock_repo_client.get_commit_file_diffs.return_value = (
            ["file1.py", "file2.py"],
            ["file3.py"],
        )
        # Create a stacktrace with files that have not changed
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file4.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file4.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files
        self.assertFalse(self.autofix_context.diff_contains_stacktrace_files(1, event_details))

    def test_diff_contains_stacktrace_files_with_removed_file(self):
        # Mock the get_commit_file_diffs method to return changed and removed files
        self.mock_repo_client.get_commit_file_diffs.return_value = (
            ["file1.py"],
            ["file2.py"],
        )
        # Create a stacktrace with a file that has been removed
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file2.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file2.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files
        self.assertTrue(self.autofix_context.diff_contains_stacktrace_files(1, event_details))

    def test_diff_contains_stacktrace_files_raises_file_not_found(self):
        # Mock the get_commit_file_diffs method to raise FileNotFoundError
        self.mock_repo_client.get_commit_file_diffs.side_effect = FileNotFoundError
        # Create a stacktrace with any file
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file1.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file1.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.autofix_context.diff_contains_stacktrace_files(1, event_details)


if __name__ == "__main__":
    unittest.main()
