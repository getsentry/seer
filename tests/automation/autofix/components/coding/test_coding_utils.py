import textwrap

import pytest

from seer.automation.autofix.components.coding.models import FuzzyDiffChunk, PlanTaskPromptXml
from seer.automation.autofix.components.coding.utils import (
    extract_diff_chunks,
    task_to_file_change,
    task_to_file_create,
    task_to_file_delete,
)
from seer.automation.models import FileChange


class TestExtractDiffChunks:
    @pytest.fixture
    def simple_diff(self):
        return """@@ -1,3 +1,3 @@
-This is the original line
+This is the modified line
 This line remains unchanged
-This line will be removed"""

    @pytest.fixture
    def complex_diff(self):
        return """@@ -1,5 +1,6 @@
 First unchanged line
-Second line to be removed
+Second line modified
+New line added
 Third unchanged line
-Fourth line to be removed
+Fourth line modified
 Fifth unchanged line
@@ -10,3 +11,4 @@
 Tenth line unchanged
-Eleventh line removed
+Eleventh line modified
+Twelfth line added"""

    def test_extract_single_chunk(self, simple_diff):
        result = extract_diff_chunks(simple_diff)
        assert len(result) == 1
        assert isinstance(result[0], FuzzyDiffChunk)
        assert result[0].header == "@@ -1,3 +1,3 @@"
        assert (
            result[0].original_chunk
            == "This is the original line\nThis line remains unchanged\nThis line will be removed"
        )
        assert result[0].new_chunk == "This is the modified line\nThis line remains unchanged"

    def test_extract_multiple_chunks(self, complex_diff):
        result = extract_diff_chunks(complex_diff)
        assert len(result) == 2

        assert result[0].header == "@@ -1,5 +1,6 @@"
        assert (
            result[0].original_chunk
            == "First unchanged line\nSecond line to be removed\nThird unchanged line\nFourth line to be removed\nFifth unchanged line"
        )
        assert (
            result[0].new_chunk
            == "First unchanged line\nSecond line modified\nNew line added\nThird unchanged line\nFourth line modified\nFifth unchanged line"
        )

        assert result[1].header == "@@ -10,3 +11,4 @@"
        assert result[1].original_chunk == "Tenth line unchanged\nEleventh line removed"
        assert (
            result[1].new_chunk
            == "Tenth line unchanged\nEleventh line modified\nTwelfth line added"
        )

    def test_empty_diff(self):
        result = extract_diff_chunks("")
        assert len(result) == 0

    def test_diff_without_changes(self):
        diff = "@@ -1,3 +1,3 @@\n Line 1\n Line 2\n Line 3"
        result = extract_diff_chunks(diff)
        assert len(result) == 1
        assert result[0].original_chunk == result[0].new_chunk == "Line 1\nLine 2\nLine 3"

    def test_diff_with_only_additions(self):
        diff = "@@ -1,1 +1,3 @@\n Unchanged line\n+Added line 1\n+Added line 2"
        result = extract_diff_chunks(diff)
        assert len(result) == 1
        assert result[0].original_chunk == "Unchanged line"
        assert result[0].new_chunk == "Unchanged line\nAdded line 1\nAdded line 2"

    def test_diff_with_only_deletions(self):
        diff = "@@ -1,3 +1,1 @@\n-Removed line 1\n-Removed line 2\n Unchanged line"
        result = extract_diff_chunks(diff)
        assert len(result) == 1
        assert result[0].original_chunk == "Removed line 1\nRemoved line 2\nUnchanged line"
        assert result[0].new_chunk == "Unchanged line"

    def test_diff_with_multiple_unchanged_lines(self):
        diff = "@@ -1,5 +1,5 @@\n Unchanged 1\n-Removed\n+Added\n Unchanged 2\n Unchanged 3\n Unchanged 4"
        result = extract_diff_chunks(diff)
        assert len(result) == 1
        assert (
            result[0].original_chunk
            == "Unchanged 1\nRemoved\nUnchanged 2\nUnchanged 3\nUnchanged 4"
        )
        assert result[0].new_chunk == "Unchanged 1\nAdded\nUnchanged 2\nUnchanged 3\nUnchanged 4"

    def test_diff_with_non_ascii_characters(self):
        diff = "@@ -1,2 +1,2 @@\n-こんにちは\n+Hello\n 世界"
        result = extract_diff_chunks(diff)
        assert len(result) == 1
        assert result[0].original_chunk == "こんにちは\n世界"
        assert result[0].new_chunk == "Hello\n世界"

    @pytest.mark.parametrize(
        "invalid_diff",
        [
            "Invalid diff content",
            "--- a/file.txt\n+++ b/file.txt\nInvalid content",
            "@ Invalid hunk header @@\nContent",
        ],
    )
    def test_still_works_with_invalid_diff_format(self, invalid_diff):
        result = extract_diff_chunks(invalid_diff)
        assert len(result) == 0

    def test_diff_with_empty_lines(self):
        diff = "@@ -1,4 +1,4 @@\n \n-Removed line\n+Added line\n \n No change"
        result = extract_diff_chunks(diff)
        assert len(result) == 1
        assert result[0].original_chunk == "\nRemoved line\n\nNo change"
        assert result[0].new_chunk == "\nAdded line\n\nNo change"


class TestTaskToFileCreate:
    def test_valid_file_create_task(self):
        task = PlanTaskPromptXml(
            file_path="new_file.py",
            repo_name="test_repo",
            type="file_create",
            diff="@@ -0,0 +1,3 @@\n+def new_function():\n+    return 'Hello, World!'\n+",
            description="Create a new file with a simple function",
            commit_message="Add new_file.py with new_function",
        )
        result = task_to_file_create(task)
        assert isinstance(result, FileChange)
        assert result.change_type == "create"
        assert result.path == "new_file.py"
        assert result.new_snippet == "def new_function():\n    return 'Hello, World!'\n"

    def test_invalid_task_type(self):
        task = PlanTaskPromptXml(
            file_path="existing_file.py",
            repo_name="test_repo",
            type="file_change",
            diff="@@ -1,1 +1,1 @@\n-old content\n+new content",
            description="Modify existing file",
            commit_message="Update existing_file.py",
        )
        with pytest.raises(ValueError, match="Expected file_create task, got: file_change"):
            task_to_file_create(task)

    def test_multiple_diff_chunks(self):
        task = PlanTaskPromptXml(
            file_path="new_file.py",
            repo_name="test_repo",
            type="file_create",
            diff="@@ -0,0 +1,2 @@\n+def func1():\n+    pass\n@@ -0,0 +3,4 @@\n+def func2():\n+    pass\n",
            description="Create a new file with two functions",
            commit_message="Add new_file.py with two functions",
        )
        with pytest.raises(ValueError, match="Expected exactly one diff chunk for file creation"):
            task_to_file_create(task)


class TestTaskToFileDelete:
    def test_valid_file_delete_task(self):
        task = PlanTaskPromptXml(
            file_path="obsolete_file.py",
            repo_name="test_repo",
            type="file_delete",
            diff="",  # No diff needed for file deletion
            description="Remove obsolete file",
            commit_message="Delete obsolete_file.py",
        )
        result = task_to_file_delete(task)
        assert isinstance(result, FileChange)
        assert result.change_type == "delete"
        assert result.path == "obsolete_file.py"

    def test_invalid_task_type(self):
        task = PlanTaskPromptXml(
            file_path="existing_file.py",
            repo_name="test_repo",
            type="file_change",
            diff="@@ -1,1 +1,1 @@\n-old content\n+new content",
            description="Modify existing file",
            commit_message="Update existing_file.py",
        )
        with pytest.raises(ValueError, match="Expected file_delete task, got: file_change"):
            task_to_file_delete(task)


class TestTaskToFileChange:
    def test_valid_file_change_task(self):
        task = PlanTaskPromptXml(
            file_path="existing_file.py",
            repo_name="test_repo",
            type="file_change",
            diff="@@ -1,3 +1,3 @@\n def existing_function():\n-    return 'Hello'\n+    return 'Hello, World!'",
            description="Update return value of existing_function",
            commit_message="Modify existing_function in existing_file.py",
        )
        file_content = "def existing_function():\n    return 'Hello'\n"
        result = task_to_file_change(task, file_content)
        assert len(result) == 1
        assert isinstance(result[0], FileChange)
        assert result[0].change_type == "edit"
        assert result[0].path == "existing_file.py"
        assert result[0].reference_snippet == "def existing_function():\n    return 'Hello'"
        assert result[0].new_snippet == "def existing_function():\n    return 'Hello, World!'"

    def test_indented_subsnippet(self):
        task = PlanTaskPromptXml(
            file_path="existing_file.py",
            repo_name="test_repo",
            type="file_change",
            diff="@@ -3,3 +3,3 @@\n     def inner_function():\n-        return 'Old'\n+        return 'New'",
            description="Update return value of inner_function",
            commit_message="Modify inner_function in existing_file.py",
        )
        file_content = textwrap.dedent(
            """\
                def outer_function():
                    # Some comment
                    def inner_function():
                        return 'Old'
                    # Another comment
                    print("Hello")
            """
        )
        result = task_to_file_change(task, file_content)
        assert len(result) == 1
        assert isinstance(result[0], FileChange)
        assert result[0].change_type == "edit"
        assert result[0].path == "existing_file.py"
        assert result[0].reference_snippet == "    def inner_function():\n        return 'Old'"
        assert result[0].new_snippet == "    def inner_function():\n        return 'New'"

    def test_invalid_task_type(self):
        task = PlanTaskPromptXml(
            file_path="new_file.py",
            repo_name="test_repo",
            type="file_create",
            diff="@@ -0,0 +1,3 @@\n+def new_function():\n+    pass\n",
            description="Create a new file",
            commit_message="Add new_file.py",
        )
        with pytest.raises(ValueError, match="Expected file_change task, got: file_create"):
            task_to_file_change(task, "")

    def test_snippet_not_found(self):
        task = PlanTaskPromptXml(
            file_path="existing_file.py",
            repo_name="test_repo",
            type="file_change",
            diff="@@ -1,3 +1,3 @@\n def non_existing_function():\n-    return 'Old'\n+    return 'New'",
            description="Update non-existing function",
            commit_message="Modify non_existing_function in existing_file.py",
        )
        file_content = "def existing_function():\n    return 'Hello'\n"
        result = task_to_file_change(task, file_content)
        assert len(result) == 0

    def test_multiple_chunks(self):
        task = PlanTaskPromptXml(
            file_path="existing_file.py",
            repo_name="test_repo",
            type="file_change",
            diff="@@ -1,3 +1,3 @@\n def func1():\n-    return 'Old1'\n+    return 'New1'\n@@ -5,3 +5,3 @@\n def func2():\n-    return 'Old2'\n+    return 'New2'",
            description="Update two functions",
            commit_message="Modify func1 and func2 in existing_file.py",
        )
        file_content = "def func1():\n    return 'Old1'\n\ndef func2():\n    return 'Old2'\n"
        result = task_to_file_change(task, file_content)
        assert len(result) == 2
        assert result[0].new_snippet == "def func1():\n    return 'New1'"
        assert result[1].new_snippet == "def func2():\n    return 'New2'"
