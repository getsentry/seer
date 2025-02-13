import pytest

from seer.automation.models import (
    FileChange,
    FileChangeError,
    FilePatch,
    Hunk,
    Line,
    Profile,
    ProfileFrame,
)


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


def test_file_patch_apply_multiple_hunks():
    patch = FilePatch(
        type="M",
        path="file.txt",
        added=2,
        removed=2,
        source_file="file.txt",
        target_file="file.txt",
        hunks=[
            Hunk(
                source_start=1,
                source_length=3,
                target_start=1,
                target_length=3,
                section_header="@@ -1,3 +1,3 @@",
                lines=[
                    Line(source_line_no=1, target_line_no=1, value="First line\n", line_type=" "),
                    Line(source_line_no=2, value="old second line\n", line_type="-"),
                    Line(target_line_no=2, value="new second line\n", line_type="+"),
                    Line(source_line_no=3, target_line_no=3, value="Third line\n", line_type=" "),
                ],
            ),
            Hunk(
                source_start=5,
                source_length=3,
                target_start=5,
                target_length=3,
                section_header="@@ -5,3 +5,3 @@",
                lines=[
                    Line(source_line_no=5, target_line_no=5, value="Fifth line\n", line_type=" "),
                    Line(source_line_no=6, value="old sixth line\n", line_type="-"),
                    Line(target_line_no=6, value="new sixth line\n", line_type="+"),
                    Line(source_line_no=7, target_line_no=7, value="Seventh line", line_type=" "),
                ],
            ),
        ],
    )

    original_content = "First line\nold second line\nThird line\nFourth line\nFifth line\nold sixth line\nSeventh line"
    result = patch.apply(original_content)
    assert (
        result
        == "First line\nnew second line\nThird line\nFourth line\nFifth line\nnew sixth line\nSeventh line"
    )


def test_file_patch_apply_with_line_number_mismatch():
    """Test that patch application correctly handles source vs target line numbers"""
    patch = FilePatch(
        type="M",
        path="file.txt",
        added=2,
        removed=1,
        source_file="file.txt",
        target_file="file.txt",
        hunks=[
            Hunk(
                source_start=2,  # Line numbers in original file
                source_length=2,
                target_start=4,  # Different line numbers in target file
                target_length=3,
                section_header="@@ -2,2 +4,3 @@",
                lines=[
                    Line(source_line_no=2, target_line_no=4, value="unchanged\n", line_type=" "),
                    Line(source_line_no=3, value="to_remove\n", line_type="-"),
                    Line(target_line_no=5, value="new_line1\n", line_type="+"),
                    Line(target_line_no=6, value="new_line2\n", line_type="+"),
                ],
            )
        ],
    )

    original_content = "line1\nunchanged\nto_remove\nline4\n"
    result = patch.apply(original_content)
    assert result == "line1\nunchanged\nnew_line1\nnew_line2\nline4\n"


def test_file_patch_apply_with_target_line_increments():
    """Test that patch application handles multiple hunks where target line numbers increase due to previous changes"""
    patch = FilePatch(
        type="M",
        path="file.txt",
        added=3,
        removed=1,
        source_file="file.txt",
        target_file="file.txt",
        hunks=[
            Hunk(
                source_start=2,
                source_length=2,
                target_start=2,
                target_length=3,
                section_header="@@ -2,2 +2,3 @@",
                lines=[
                    Line(source_line_no=2, target_line_no=2, value="unchanged1\n", line_type=" "),
                    Line(source_line_no=3, value="to_remove1\n", line_type="-"),
                    Line(target_line_no=3, value="new_line1\n", line_type="+"),
                    Line(target_line_no=4, value="new_line2\n", line_type="+"),
                ],
            ),
            Hunk(
                source_start=5,  # Original source line
                source_length=2,
                target_start=6,  # Target line increased by 1 due to previous hunk adding a line
                target_length=3,
                section_header="@@ -5,2 +6,3 @@",
                lines=[
                    Line(source_line_no=5, target_line_no=6, value="unchanged2\n", line_type=" "),
                    Line(source_line_no=6, value="to_remove2\n", line_type="-"),
                    Line(target_line_no=7, value="new_line3\n", line_type="+"),
                    Line(target_line_no=8, value="new_line4\n", line_type="+"),
                ],
            ),
        ],
    )

    original_content = "line1\nunchanged1\nto_remove1\nline4\nunchanged2\nto_remove2\nline7\n"
    result = patch.apply(original_content)
    assert (
        result
        == "line1\nunchanged1\nnew_line1\nnew_line2\nline4\nunchanged2\nnew_line3\nnew_line4\nline7\n"
    )


def test_file_patch_apply_raises_on_hunk_error():
    """Test that patch.apply raises FileChangeError when _apply_hunks fails"""
    patch = FilePatch(
        type="M",
        path="file.txt",
        added=1,
        removed=1,
        source_file="file.txt",
        target_file="file.txt",
        hunks=[
            Hunk(
                source_start=8,  # Source line that doesn't exist
                source_length=2,
                target_start=2,
                target_length=2,
                section_header="@@ -2,2 +2,2 @@",
                lines=[
                    Line(source_line_no=2, target_line_no=2, value="unchanged\n", line_type=" "),
                    Line(source_line_no=3, value="old\n", line_type="-"),
                    Line(target_line_no=3, value="new\n", line_type="+"),
                ],
            ),
        ],
    )

    original_content = "line1\ndifferent_content\nold\nline4\n"

    with pytest.raises(FileChangeError, match="Error applying hunks"):
        patch.apply(original_content)


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


def test_profile_format_no_relevant_functions():
    """Test that profile formatting works without relevant functions"""
    profile = Profile(
        profile_matches_issue=True,
        execution_tree=[
            ProfileFrame(
                function="main",
                module="app",
                filename="app.py",
                lineno=1,
                in_app=True,
                children=[
                    ProfileFrame(
                        function="helper",
                        module="utils",
                        filename="utils.py",
                        lineno=5,
                        in_app=True,
                    )
                ],
            )
        ],
    )

    expected = "→ main (app.py)\n  → helper (utils.py)"
    assert profile.format_profile() == expected


def test_profile_format_with_relevant_functions():
    """Test that profile formatting focuses on relevant functions"""
    profile = Profile(
        profile_matches_issue=True,
        execution_tree=[
            ProfileFrame(
                function="main",
                module="app",
                filename="app.py",
                lineno=1,
                in_app=True,
                children=[
                    ProfileFrame(
                        function="process_data",
                        module="utils",
                        filename="utils.py",
                        lineno=5,
                        in_app=True,
                    ),
                    ProfileFrame(
                        function="relevant_func",
                        module="core",
                        filename="core.py",
                        lineno=10,
                        in_app=True,
                    ),
                    ProfileFrame(
                        function="cleanup",
                        module="utils",
                        filename="utils.py",
                        lineno=15,
                        in_app=True,
                    ),
                ],
            )
        ],
        relevant_functions={"relevant_func"},
    )

    # Should only show the relevant function with context
    formatted = profile.format_profile(context_before=1, context_after=1)
    expected = (
        "...\n  → process_data (utils.py)\n  → relevant_func (core.py)\n  → cleanup (utils.py)"
    )
    assert formatted == expected


def test_profile_format_with_multiple_relevant_functions():
    """Test that profile formatting handles multiple relevant functions"""
    profile = Profile(
        profile_matches_issue=True,
        execution_tree=[
            ProfileFrame(
                function="main",
                module="app",
                filename="app.py",
                lineno=1,
                in_app=True,
                children=[
                    ProfileFrame(
                        function="relevant_func1",
                        module="core",
                        filename="core.py",
                        lineno=5,
                        in_app=True,
                    ),
                    ProfileFrame(
                        function="process_data",
                        module="utils",
                        filename="utils.py",
                        lineno=10,
                        in_app=True,
                    ),
                    ProfileFrame(
                        function="relevant_func2",
                        module="core",
                        filename="core.py",
                        lineno=15,
                        in_app=True,
                    ),
                ],
            )
        ],
        relevant_functions={"relevant_func1", "relevant_func2"},
    )

    # Should show both relevant functions with context
    formatted = profile.format_profile(context_before=1, context_after=1)
    expected = "→ main (app.py)\n  → relevant_func1 (core.py)\n  → process_data (utils.py)\n  → relevant_func2 (core.py)"
    assert formatted == expected


def test_profile_format_with_nested_relevant_functions():
    """Test that profile formatting works with nested relevant functions"""
    profile = Profile(
        profile_matches_issue=True,
        execution_tree=[
            ProfileFrame(
                function="main",
                module="app",
                filename="app.py",
                lineno=1,
                in_app=True,
                children=[
                    ProfileFrame(
                        function="outer",
                        module="core",
                        filename="core.py",
                        lineno=5,
                        in_app=True,
                        children=[
                            ProfileFrame(
                                function="relevant_func",
                                module="core",
                                filename="core.py",
                                lineno=10,
                                in_app=True,
                            )
                        ],
                    ),
                ],
            )
        ],
        relevant_functions={"relevant_func"},
    )

    # Should show the nested structure leading to the relevant function
    formatted = profile.format_profile(context_before=2, context_after=0)
    expected = "→ main (app.py)\n  → outer (core.py)\n    → relevant_func (core.py)"
    assert formatted == expected


def test_profile_format_with_custom_context():
    """Test that profile formatting respects custom context sizes"""
    profile = Profile(
        profile_matches_issue=True,
        execution_tree=[
            ProfileFrame(
                function="main",
                module="app",
                filename="app.py",
                lineno=1,
                in_app=True,
                children=[
                    ProfileFrame(
                        function="func1",
                        module="utils",
                        filename="utils.py",
                        lineno=5,
                        in_app=True,
                    ),
                    ProfileFrame(
                        function="relevant_func",
                        module="core",
                        filename="core.py",
                        lineno=10,
                        in_app=True,
                    ),
                    ProfileFrame(
                        function="func2",
                        module="utils",
                        filename="utils.py",
                        lineno=15,
                        in_app=True,
                    ),
                ],
            )
        ],
        relevant_functions={"relevant_func"},
    )

    # Test with minimal context
    minimal_context = profile.format_profile(context_before=0, context_after=0)
    assert minimal_context == "...\n  → relevant_func (core.py)\n..."

    # Test with asymmetric context
    asymmetric_context = profile.format_profile(context_before=1, context_after=0)
    assert asymmetric_context == "...\n  → func1 (utils.py)\n  → relevant_func (core.py)\n..."


def test_stacktraceframe_filtering():
    from seer.automation.models import StacktraceFrame

    # Test _contains_filtered
    assert not StacktraceFrame._contains_filtered("This is safe")
    assert StacktraceFrame._contains_filtered("This is [Filtered] content")

    # Test _filter_nested_value with a nested dict and list structure
    sample_dict = {
        "key1": "safe value",
        "key2": "this is [Filtered] value",
        "key3": {
            "subkey1": "another [Filtered] test",
            "subkey2": "clean value",
        },
        "key4": [
            "value1",
            "value2",
            "value [Filtered] extra",
            {
                "list_key": "normal",
                "filter": "[Filtered]",
            },
        ],
    }
    filtered_dict = StacktraceFrame._filter_nested_value(sample_dict)
    expected_filtered_dict = {
        "key1": "safe value",
        "key3": {"subkey2": "clean value"},
        "key4": ["value1", "value2", {"list_key": "normal"}],
    }
    assert filtered_dict == expected_filtered_dict

    # Test _trim_vars which should only keep keys mentioned in code_context and also apply filtering
    vars_input = {
        "key1": "safe value",
        "key2": "this is [Filtered] value",
        "key3": {
            "subkey1": "another [Filtered] test",
            "subkey2": "clean value",
            "subkey3": "extra value",
        },
        "key4": [
            "value1",
            "value2",
            "value [Filtered] extra",
            {"list_key": "normal", "filter": "[Filtered]"},
        ],
        "key5": "not mentioned",
    }
    # Only keys "key1", "key3", and "key4" are mentioned in the code context.
    code_context = "key1 key3 key4"
    trimmed_vars = StacktraceFrame._trim_vars(vars_input, code_context)
    expected_trimmed_vars = {
        "key1": "safe value",
        "key3": {"subkey2": "clean value", "subkey3": "extra value"},
        "key4": ["value1", "value2", {"list_key": "normal"}],
    }
    assert trimmed_vars == expected_trimmed_vars
