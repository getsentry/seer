import pytest

from seer.automation.models import FileChange, FileChangeError, FilePatch, Hunk, Line


def test_file_patch_apply_add():
    patch = FilePatch(
        type="A",
        path="new_file.txt",
        added=3,
        removed=0,
        source_file="",
        target_file="new_file.txt",
        hunks=[
            Hunk(
                source_start=0,
                source_length=0,
                target_start=1,
                target_length=3,
                section_header="@@ -0,0 +1,3 @@",
                lines=[
                    Line(target_line_no=1, value="Line 1\n", line_type="+"),
                    Line(target_line_no=2, value="Line 2\n", line_type="+"),
                    Line(target_line_no=3, value="Line 3", line_type="+"),
                ],
            )
        ],
    )

    result = patch.apply(None)
    assert result == "Line 1\nLine 2\nLine 3"

    with pytest.raises(FileChangeError):
        patch.apply("Existing content")


def test_file_patch_apply_modify():
    patch = FilePatch(
        type="M",
        path="existing_file.txt",
        added=1,
        removed=1,
        source_file="existing_file.txt",
        target_file="existing_file.txt",
        hunks=[
            Hunk(
                source_start=2,
                source_length=3,
                target_start=2,
                target_length=3,
                section_header="@@ -2,3 +2,3 @@",
                lines=[
                    Line(source_line_no=2, target_line_no=2, value="Line 2\n", line_type=" "),
                    Line(source_line_no=3, value="Old Line 3\n", line_type="-"),
                    Line(target_line_no=3, value="New Line 3\n", line_type="+"),
                    Line(source_line_no=4, target_line_no=4, value="Line 4", line_type=" "),
                ],
            )
        ],
    )

    original_content = "Line 1\nLine 2\nOld Line 3\nLine 4\nLine 5"
    result = patch.apply(original_content)
    assert result == "Line 1\nLine 2\nNew Line 3\nLine 4\nLine 5"

    with pytest.raises(FileChangeError):
        patch.apply(None)


def test_file_patch_apply_delete():
    patch = FilePatch(
        type="D",
        path="file_to_delete.txt",
        added=0,
        removed=3,
        source_file="file_to_delete.txt",
        target_file="",
        hunks=[],
    )

    result = patch.apply("Content to be deleted")
    assert result is None

    with pytest.raises(FileChangeError):
        patch.apply(None)


def test_file_patch_apply_complex_modify():
    patch = FilePatch(
        type="M",
        path="complex_file.txt",
        added=2,
        removed=1,
        source_file="complex_file.txt",
        target_file="complex_file.txt",
        hunks=[
            Hunk(
                source_start=1,
                source_length=5,
                target_start=1,
                target_length=6,
                section_header="@@ -1,5 +1,6 @@",
                lines=[
                    Line(source_line_no=1, target_line_no=1, value="Line 1\n", line_type=" "),
                    Line(target_line_no=2, value="New Line 2\n", line_type="+"),
                    Line(source_line_no=2, target_line_no=3, value="Line 2\n", line_type=" "),
                    Line(source_line_no=3, value="Old Line 3\n", line_type="-"),
                    Line(target_line_no=4, value="Updated Line 3\n", line_type="+"),
                    Line(source_line_no=4, target_line_no=5, value="Line 4\n", line_type=" "),
                    Line(source_line_no=5, target_line_no=6, value="Line 5", line_type=" "),
                ],
            )
        ],
    )

    original_content = "Line 1\nLine 2\nOld Line 3\nLine 4\nLine 5"
    result = patch.apply(original_content)
    assert result == "Line 1\nNew Line 2\nLine 2\nUpdated Line 3\nLine 4\nLine 5"


def test_file_patch_apply_preserve_trailing_newlines():
    patch = FilePatch(
        type="M",
        path="file.txt",
        added=1,
        removed=1,
        source_file="file.txt",
        target_file="file.txt",
        hunks=[
            Hunk(
                source_start=1,
                source_length=2,
                target_start=1,
                target_length=2,
                section_header="@@ -1,2 +1,2 @@",
                lines=[
                    Line(source_line_no=1, target_line_no=1, value="First line\n", line_type=" "),
                    Line(source_line_no=2, value="old line\n", line_type="-"),
                    Line(target_line_no=2, value="new line\n", line_type="+"),
                ],
            )
        ],
    )

    original_content = "First line\nold line\n\n\n"  # Content with trailing newlines
    result = patch.apply(original_content)
    assert result == "First line\nnew line\n\n\n"  # Trailing newlines preserved


# New tests for FileChange


def test_file_change_create():
    change = FileChange(
        change_type="create",
        path="new_file.txt",
        new_snippet="This is a new file content.\nIt has multiple lines.",
        description="Creating a new file",
        commit_message="Add new_file.txt",
    )

    result = change.apply(None)
    assert result == "This is a new file content.\nIt has multiple lines."

    with pytest.raises(FileChangeError):
        change.apply("Existing content")


def test_file_change_create_without_new_snippet():
    change = FileChange(
        change_type="create",
        path="new_file.txt",
        description="Creating a new file",
        commit_message="Add new_file.txt",
    )

    with pytest.raises(FileChangeError):
        change.apply(None)


def test_file_change_edit():
    change = FileChange(
        change_type="edit",
        path="existing_file.txt",
        reference_snippet="old content",
        new_snippet="new content",
        description="Updating existing file",
        commit_message="Update existing_file.txt",
    )

    original_content = "This is the old content in the file."
    result = change.apply(original_content)
    assert result == "This is the new content in the file."

    with pytest.raises(FileChangeError):
        change.apply(None)


def test_file_change_edit_without_reference_snippet():
    change = FileChange(
        change_type="edit",
        path="existing_file.txt",
        new_snippet="new content",
        description="Updating existing file",
        commit_message="Update existing_file.txt",
    )

    with pytest.raises(FileChangeError):
        change.apply("Some content")


def test_file_change_edit_without_new_snippet():
    change = FileChange(
        change_type="edit",
        path="existing_file.txt",
        reference_snippet="old content",
        description="Updating existing file",
        commit_message="Update existing_file.txt",
    )

    with pytest.raises(FileChangeError):
        change.apply("Some content")


def test_file_change_delete():
    change = FileChange(
        change_type="delete",
        path="file_to_delete.txt",
        reference_snippet="content to delete",
        description="Deleting a file",
        commit_message="Remove file_to_delete.txt",
    )

    original_content = "This is the content to delete in the file."
    result = change.apply(original_content)
    assert result == "This is the  in the file."

    # Test complete file deletion
    change_full_delete = FileChange(
        change_type="delete",
        path="file_to_delete.txt",
        description="Deleting entire file",
        commit_message="Remove file_to_delete.txt",
    )
    result_full_delete = change_full_delete.apply("Any content")
    assert result_full_delete is None

    with pytest.raises(FileChangeError):
        change.apply(None)
