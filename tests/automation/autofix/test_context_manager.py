import unittest
from unittest.mock import MagicMock

from seer.automation.autofix.context_manager import ContextManager
from seer.automation.autofix.models import Stacktrace, StacktraceFrame


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.repo_owner = "seer"
        self.repo_name = "automation"
        self.base_sha = "abc123"
        self.mock_repo_client = MagicMock()
        self.context_manager = ContextManager(self.mock_repo_client, self.base_sha)

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
                    filename="file2.py", col_no=0, line_no=10, function="test", context=[]
                )
            ]
        )
        # Check if the diff contains stacktrace files
        self.assertTrue(self.context_manager.diff_contains_stacktrace_files(stacktrace))

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
                    filename="file4.py", col_no=0, line_no=10, function="test", context=[]
                )
            ]
        )
        # Check if the diff contains stacktrace files
        self.assertFalse(self.context_manager.diff_contains_stacktrace_files(stacktrace))

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
                    filename="file2.py", col_no=0, line_no=10, function="test", context=[]
                )
            ]
        )
        # Check if the diff contains stacktrace files
        self.assertTrue(self.context_manager.diff_contains_stacktrace_files(stacktrace))

    def test_diff_contains_stacktrace_files_raises_file_not_found(self):
        # Mock the get_commit_file_diffs method to raise FileNotFoundError
        self.mock_repo_client.get_commit_file_diffs.side_effect = FileNotFoundError
        # Create a stacktrace with any file
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file1.py", col_no=0, line_no=10, function="test", context=[]
                )
            ]
        )
        # Check if the diff contains stacktrace files raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.context_manager.diff_contains_stacktrace_files(stacktrace)


if __name__ == "__main__":
    unittest.main()
