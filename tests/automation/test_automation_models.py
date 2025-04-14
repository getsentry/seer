import datetime
import textwrap

import pytest

from seer.automation.models import (
    EAPTrace,
    FileChange,
    FileChangeError,
    FilePatch,
    Hunk,
    Line,
    Profile,
    ProfileFrame,
    Span,
    TraceEvent,
    TraceTree,
    format_annotated_hunks,
    right_justified,
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

    expected = "└─ main (app.py:1)\n   └─ helper (utils.py:5)"
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
        "...\n"
        "   ├─ process_data (utils.py:5)\n"
        "   ├─ relevant_func (core.py:10)\n"
        "   └─ cleanup (utils.py:15)"
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
    expected = (
        "└─ main (app.py:1)\n"
        "   ├─ relevant_func1 (core.py:5)\n"
        "   ├─ process_data (utils.py:10)\n"
        "   └─ relevant_func2 (core.py:15)"
    )
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
    expected = (
        "└─ main (app.py:1)\n" "   └─ outer (core.py:5)\n" "      └─ relevant_func (core.py:10)"
    )
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
    assert minimal_context == "...\n   ├─ relevant_func (core.py:10)\n..."

    # Test with asymmetric context
    asymmetric_context = profile.format_profile(context_before=1, context_after=0)
    assert (
        asymmetric_context == "...\n   ├─ func1 (utils.py:5)\n   ├─ relevant_func (core.py:10)\n..."
    )


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


def test_trace_tree_format_empty():
    """Test formatting an empty trace tree"""
    trace_tree = TraceTree(trace_id="empty-trace")
    formatted = trace_tree.format_trace_tree()
    assert formatted == "Trace (empty)"


def test_trace_tree_format_simple():
    """Test formatting a simple trace tree with one event"""
    trace_tree = TraceTree(
        trace_id="simple-trace",
        events=[
            TraceEvent(
                event_id="abcdef1234567890abcdef1234567890",
                title="My Transaction",
                is_transaction=True,
                platform="python",
                duration="200ms",
            )
        ],
    )

    formatted = trace_tree.format_trace_tree()
    expected = "Trace\n└─ My Transaction (200ms) (event ID: abcdef1) (python)"
    assert formatted == expected


def test_trace_tree_format_complex():
    """Test formatting a complex trace tree with nested events and all attributes"""
    trace_tree = TraceTree(
        trace_id="complex-trace",
        events=[
            TraceEvent(
                event_id="abcdef1234567890abcdef1234567890",
                title="Root Transaction",
                is_transaction=True,
                platform="python",
                duration="500ms",
                profile_id="profile1234567890abcdef",
                project_slug="main-project",
                project_id=12345,
                children=[
                    TraceEvent(
                        event_id="bcdef1234567890abcdef1234567891",
                        title="Child Transaction",
                        is_transaction=True,
                        platform="javascript",
                        duration="300ms",
                        children=[
                            TraceEvent(
                                event_id="cdef1234567890abcdef1234567892",
                                title="Error Event",
                                is_error=True,
                                platform="javascript",
                            ),
                        ],
                    ),
                    TraceEvent(
                        event_id="defgh1234567890abcdef1234567893",
                        title="External Service",
                        is_transaction=True,
                        platform="go",
                        duration="150ms",
                        is_current_project=False,
                        project_slug="external-project",
                        project_id=67890,
                    ),
                ],
            )
        ],
    )

    formatted = trace_tree.format_trace_tree()
    expected = (
        "Trace\n"
        "└─ Root Transaction (500ms) (event ID: abcdef1) (project: main-project) (python) (profile available)\n"
        "   ├─ Child Transaction (300ms) (event ID: bcdef12) (javascript)\n"
        "   │  └─ ERROR: Error Event (event ID: cdef123) (javascript)\n"
        "   └─ External Service (150ms) (event ID: defgh12) (project: external-project) (go)"
    )
    assert formatted == expected


def test_trace_tree_format_with_repetition():
    """Test formatting a trace tree with repeated events"""
    trace_tree = TraceTree(
        trace_id="repeat-trace",
        events=[
            TraceEvent(
                event_id="abcdef1234567890abcdef1234567890",
                title="API Request",
                is_transaction=True,
                platform="python",
                duration="100ms",
                children=[
                    # Three identical DB queries in sequence
                    TraceEvent(
                        event_id="dbquery1234567890abcdef1234567891",
                        title="DB Query",
                        is_transaction=True,
                        platform="sql",
                        duration="20ms",
                    ),
                    TraceEvent(
                        event_id="dbquery2234567890abcdef1234567892",
                        title="DB Query",
                        is_transaction=True,
                        platform="sql",
                        duration="20ms",
                    ),
                    TraceEvent(
                        event_id="dbquery3234567890abcdef1234567893",
                        title="DB Query",
                        is_transaction=True,
                        platform="sql",
                        duration="20ms",
                    ),
                    # Different event after the repeats
                    TraceEvent(
                        event_id="render1234567890abcdef1234567894",
                        title="Render Template",
                        is_transaction=True,
                        platform="python",
                        duration="30ms",
                    ),
                ],
            )
        ],
    )

    formatted = trace_tree.format_trace_tree()
    expected = (
        "Trace\n"
        "└─ API Request (100ms) (event ID: abcdef1) (python)\n"
        "   ├─ DB Query (20ms) (event ID: dbquery) (sql) (repeated 3 times)\n"
        "   └─ Render Template (30ms) (event ID: render1) (python)"
    )
    assert formatted == expected


def test_trace_tree_id_lookup():
    """Test the ID lookup functions in TraceTree"""
    full_event_id = "abcdef1234567890abcdef1234567890"
    full_profile_id = "profile1234567890abcdef"

    trace_tree = TraceTree(
        trace_id="id-lookup-trace",
        events=[
            TraceEvent(
                event_id=full_event_id,
                title="Transaction",
                is_transaction=True,
                profile_id=full_profile_id,
            )
        ],
    )

    # Test event ID lookup
    assert trace_tree.get_full_event_id("abcdef1") == full_event_id
    assert trace_tree.get_full_event_id("xyz123") is None

    # Test getting full event object by ID
    event = trace_tree.get_event_by_id("abcdef1")
    assert event is not None
    assert event.event_id == full_event_id
    assert event.title == "Transaction"
    assert event.is_transaction is True
    assert event.profile_id == full_profile_id

    # Test with non-existent ID
    assert trace_tree.get_event_by_id("xyz123") is None


def test_trace_tree_nested_with_repetition():
    """Test a more complex trace tree with nested repetitions"""
    trace_tree = TraceTree(
        trace_id="complex-repeat-trace",
        events=[
            TraceEvent(
                event_id="root1234567890abcdef1234567890",
                title="Web Request",
                is_transaction=True,
                platform="python",
                duration="800ms",
                children=[
                    # Two identical auth checks
                    TraceEvent(
                        event_id="auth1234567890abcdef1234567891",
                        title="Auth Check",
                        is_transaction=True,
                        platform="python",
                        duration="50ms",
                    ),
                    TraceEvent(
                        event_id="auth2234567890abcdef1234567892",
                        title="Auth Check",
                        is_transaction=True,
                        platform="python",
                        duration="50ms",
                    ),
                    # Service call with nested repeats
                    TraceEvent(
                        event_id="svc1234567890abcdef1234567893",
                        title="Service Call",
                        is_transaction=True,
                        platform="python",
                        duration="400ms",
                        children=[
                            # Three identical cache lookups
                            TraceEvent(
                                event_id="cache1234567890abcdef1234567894",
                                title="Cache Lookup",
                                is_transaction=True,
                                platform="redis",
                                duration="10ms",
                            ),
                            TraceEvent(
                                event_id="cache2234567890abcdef1234567895",
                                title="Cache Lookup",
                                is_transaction=True,
                                platform="redis",
                                duration="10ms",
                            ),
                            TraceEvent(
                                event_id="cache3234567890abcdef1234567896",
                                title="Cache Lookup",
                                is_transaction=True,
                                platform="redis",
                                duration="10ms",
                            ),
                            # One error at the same level
                            TraceEvent(
                                event_id="error1234567890abcdef1234567897",
                                title="Database Error",
                                is_error=True,
                                platform="postgresql",
                            ),
                        ],
                    ),
                ],
            )
        ],
    )

    formatted = trace_tree.format_trace_tree()
    expected = (
        "Trace\n"
        "└─ Web Request (800ms) (event ID: root123) (python)\n"
        "   ├─ Auth Check (50ms) (event ID: auth123) (python) (repeated 2 times)\n"
        "   └─ Service Call (400ms) (event ID: svc1234) (python)\n"
        "      ├─ Cache Lookup (10ms) (event ID: cache12) (redis) (repeated 3 times)\n"
        "      └─ ERROR: Database Error (event ID: error12) (postgresql)"
    )
    assert formatted == expected


def test_trace_event_format_spans_tree():
    """Test formatting a spans tree with mixed patterns and special characters"""
    event = TraceEvent(
        event_id="event123",
        title="Transaction with mixed spans",
        is_transaction=True,
        spans=[
            Span(
                span_id="http1",
                title="HTTP POST /api/submit",
                duration="850ms",
                data={"method": "POST", "path": "/api/submit", "status_code": 201},
                children=[
                    Span(
                        span_id="auth1",
                        title="JWT Validation",
                        duration="15ms",
                        data={"user_id": 12345, "scopes": ["read", "write"]},
                    ),
                    Span(
                        span_id="db1",
                        title="SELECT FROM users",
                        duration="25ms",
                        data={"db": "postgres", "rows": 1},
                    ),
                    Span(
                        span_id="db2",
                        title="SELECT FROM users",
                        duration="25ms",
                        data={"db": "postgres", "rows": 1},
                    ),
                    Span(
                        span_id="db3",
                        title="INSERT INTO events",
                        duration="45ms",
                        data={"db": "postgres", "affected_rows": 1},
                    ),
                    Span(
                        span_id="special1",
                        title="Process data: user=jsmith&type=admin",
                        duration="120ms",
                    ),
                    Span(
                        span_id="cache1",
                        title="Redis Cache",
                        duration="75ms",
                        data={"cache": "redis", "key": "user:12345"},
                        children=[
                            Span(span_id="cache_op1", title="Connect", duration="5ms"),
                            Span(span_id="cache_op2", title="Get", duration="8ms"),
                            Span(span_id="cache_op3", title="Get", duration="8ms"),
                            Span(span_id="cache_op4", title="Set", duration="12ms"),
                        ],
                    ),
                ],
            ),
            Span(
                span_id="resp1",
                title="Format Response",
                duration="35ms",
                data={"format": "JSON", "size": "24.5KB"},
            ),
        ],
    )

    formatted = event.format_spans_tree()
    expected = (
        "Spans for Transaction with mixed spans\n"
        "├─ HTTP POST /api/submit (850ms)\n"
        "│   {\n"
        '│     "method": "POST",\n'
        '│     "path": "/api/submit",\n'
        '│     "status_code": 201\n'
        "│   }\n"
        "│  ├─ JWT Validation (15ms)\n"
        "│  │   {\n"
        '│  │     "user_id": 12345,\n'
        '│  │     "scopes": [\n'
        '│  │       "read",\n'
        '│  │       "write"\n'
        "│  │     ]\n"
        "│  │   }\n"
        "│  ├─ SELECT FROM users (25ms) (repeated 2 times)\n"
        "│  │   {\n"
        '│  │     "db": "postgres",\n'
        '│  │     "rows": 1\n'
        "│  │   }\n"
        "│  ├─ INSERT INTO events (45ms)\n"
        "│  │   {\n"
        '│  │     "db": "postgres",\n'
        '│  │     "affected_rows": 1\n'
        "│  │   }\n"
        "│  ├─ Process data: user=jsmith&type=admin (120ms)\n"
        "│  └─ Redis Cache (75ms)\n"
        "│      {\n"
        '│        "cache": "redis",\n'
        '│        "key": "user:12345"\n'
        "│      }\n"
        "│     ├─ Connect (5ms)\n"
        "│     ├─ Get (8ms) (repeated 2 times)\n"
        "│     └─ Set (12ms)\n"
        "└─ Format Response (35ms)\n"
        "    {\n"
        '      "format": "JSON",\n'
        '      "size": "24.5KB"\n'
        "    }"
    )
    assert formatted == expected


def test_eap_trace_basic_creation():
    """Test basic creation of an EAPTrace object"""
    trace_data = [
        {"id": "span1", "name": "Main Transaction", "is_transaction": True, "children": []}
    ]

    trace = EAPTrace(trace_id="trace-123", trace=trace_data, timestamp=datetime.datetime.now())

    assert trace.trace_id == "trace-123"
    assert trace.trace == trace_data


def test_eap_trace_get_transaction_spans_empty():
    """Test _get_transaction_spans with empty trace"""
    trace = EAPTrace(trace_id="empty-trace", trace=[], timestamp=datetime.datetime.now())

    # Test empty trace
    result = trace.get_and_format_trace()
    assert result == ""


def test_get_and_format_trace():
    """Test _get_transaction_spans with a simple trace"""

    trace_data = [
        {"id": "span1", "name": "Transaction 1", "is_transaction": True, "children": []},
        {
            "id": "span2",
            "name": "Non-Transaction Span",
            "is_transaction": False,
            "children": [
                {
                    "id": "span2_1",
                    "name": "Non-Transaction Span 1",
                    "is_transaction": False,
                    "children": [],
                },
                {
                    "id": "span2_2",
                    "name": "Non-Transaction Span 2",
                    "is_transaction": False,
                    "children": [],
                },
            ],
        },
        {"id": "span3", "name": "Transaction 2", "is_transaction": True, "children": []},
    ]

    trace = EAPTrace(trace_id="simple-trace", trace=trace_data, timestamp=datetime.datetime.now())

    result = trace.get_and_format_trace()
    expected = """<txn id="span1" name="Transaction 1" is_transaction="True" />
<span id="span2" name="Non-Transaction Span" is_transaction="False">
    <span id="span2_1" name="Non-Transaction Span 1" is_transaction="False" />
    <span id="span2_2" name="Non-Transaction Span 2" is_transaction="False" />
</span>
<txn id="span3" name="Transaction 2" is_transaction="True" />"""
    assert result == expected

    trace_data = [
        {"id": "span1", "name": "Transaction 1", "is_transaction": True, "children": []},
        {"id": "span2", "name": "Non-Transaction Span", "is_transaction": False, "children": []},
        {"id": "span3", "name": "Transaction 2", "is_transaction": True, "children": []},
    ]

    trace = EAPTrace(trace_id="simple-trace", trace=trace_data, timestamp=datetime.datetime.now())

    result = trace.get_and_format_trace()
    expected = """<txn id="span1" name="Transaction 1" is_transaction="True" />
<span id="span2" name="Non-Transaction Span" is_transaction="False" />
<txn id="span3" name="Transaction 2" is_transaction="True" />"""
    assert result == expected

    result = trace.get_and_format_trace(only_transactions=True)
    expected = """<txn id="span1" name="Transaction 1" is_transaction="True" />\n<txn id="span3" name="Transaction 2" is_transaction="True" />"""
    assert result == expected


class TestRightJustified:
    def test_single_digit(self):
        result = right_justified(1, 5)
        assert result == ["1", "2", "3", "4", "5"]

    def test_double_digit(self):
        result = right_justified(5, 15)
        assert result == [" 5", " 6", " 7", " 8", " 9", "10", "11", "12", "13", "14", "15"]

    def test_triple_digit(self):
        result = right_justified(95, 105)
        assert result == [
            " 95",
            " 96",
            " 97",
            " 98",
            " 99",
            "100",
            "101",
            "102",
            "103",
            "104",
            "105",
        ]


@pytest.fixture
def patch_and_hunks() -> tuple[str, list[Hunk]]:
    patch = textwrap.dedent(
        """\
        @@ -1,3 +1,4 @@
         def hello():
             print('hello')
        +    print('world')  # Line 3 is added
             print('goodbye')
        __WHITESPACE__
        @@ -20,2 +21,3 @@ def __init__(self):
             print('end')
        +    print('new end')  # Line 22 is added
             return"""
    ).replace("__WHITESPACE__", " ")

    hunks = [
        Hunk(
            source_start=1,
            source_length=3,
            target_start=1,
            target_length=4,
            section_header="@@ -1,3 +1,4 @@",
            lines=[
                Line(
                    source_line_no=1,
                    target_line_no=1,
                    line_type=" ",
                    value=" def hello():",
                ),
                Line(
                    source_line_no=2,
                    target_line_no=2,
                    line_type=" ",
                    value="     print('hello')",
                ),
                Line(
                    source_line_no=None,
                    target_line_no=3,
                    line_type="+",
                    value="+    print('world')  # Line 3 is added",
                ),
                Line(
                    source_line_no=3,
                    target_line_no=4,
                    line_type=" ",
                    value="     print('goodbye')",
                ),
                Line(
                    source_line_no=4,
                    target_line_no=5,
                    line_type=" ",
                    value=" ",
                ),
            ],
        ),
        Hunk(
            source_start=20,
            source_length=2,
            target_start=21,
            target_length=3,
            section_header="@@ -20,2 +21,3 @@ def __init__(self):",
            lines=[
                Line(
                    source_line_no=20,
                    target_line_no=21,
                    line_type=" ",
                    value="     print('end')",
                ),
                Line(
                    source_line_no=None,
                    target_line_no=22,
                    line_type="+",
                    value="+    print('new end')  # Line 22 is added",
                ),
                Line(
                    source_line_no=21,
                    target_line_no=23,
                    line_type=" ",
                    value="     return",
                ),
            ],
        ),
    ]

    return patch, hunks


def test_file_patch_to_hunks(patch_and_hunks: tuple[str, list[Hunk]]):
    patch, hunks_expected = patch_and_hunks
    hunks = FilePatch.to_hunks(patch)
    assert len(hunks) == len(hunks_expected)
    assert hunks == hunks_expected

    assert "\n".join(hunk.raw() for hunk in hunks) == patch

    num_lines_hunks = [len(hunk.annotated().split("\n")) for hunk in hunks]
    assert num_lines_hunks == [len(hunk.lines) + 1 for hunk in hunks_expected]
    # +1 for the section header

    annotated_hunk = hunks[0].annotated()
    annotated_hunk_expected = textwrap.dedent(
        """\
                @@ -1,3 +1,4 @@
        1    1  def hello():
        2    2      print('hello')
             3 +    print('world')  # Line 3 is added
        3    4      print('goodbye')
        4    5  """
    )
    assert annotated_hunk == annotated_hunk_expected


def test_format_annotated_hunks(patch_and_hunks: tuple[str, list[Hunk]]):
    _, hunks = patch_and_hunks
    annotated_hunks = format_annotated_hunks(hunks)
    annotated_hunks_expected = textwrap.dedent(
        """\
                  @@ -1,3 +1,4 @@
         1     1  def hello():
         2     2      print('hello')
               3 +    print('world')  # Line 3 is added
         3     4      print('goodbye')
         4     5  __NO_SPACE__
        __NO_SPACE__
                  @@ -20,2 +21,3 @@ def __init__(self):
        20    21      print('end')
              22 +    print('new end')  # Line 22 is added
        21    23      return"""
    ).replace("__NO_SPACE__", "")
    assert annotated_hunks == annotated_hunks_expected
