import pytest

from seer.automation.codebase.utils import code_snippet, potential_frame_match, right_justified
from seer.automation.models import StacktraceFrame


class TestPotentialFrameMatch:
    def test_simple_case(self):
        frame = StacktraceFrame(
            filename="src/seer/automation/codebase/utils.py",
            abs_path="",
            line_no=1,
            col_no=1,
            context=[],
        )

        assert potential_frame_match("/home/app/src/seer/automation/codebase/utils.py", frame)

    def test_relative_path(self):
        frame = StacktraceFrame(
            filename="./src/seer/automation/codebase/utils.py",
            abs_path="",
            line_no=1,
            col_no=1,
            context=[],
        )

        assert potential_frame_match("app/src/seer/automation/codebase/utils.py", frame)

    def test_relative_parent_path(self):
        frame = StacktraceFrame(
            filename="../src/seer/automation/codebase/utils.py",
            abs_path="",
            line_no=1,
            col_no=1,
            context=[],
        )

        assert potential_frame_match("app/src/seer/automation/codebase/utils.py", frame)

    def test_non_matching_path(self):
        frame = StacktraceFrame(
            filename="src/seer/automation/codebase/utils.py",
            abs_path="",
            line_no=1,
            col_no=1,
            context=[],
        )

        assert not (
            potential_frame_match("/home/app/src/seer/automation/codebase/test_utils.py", frame)
        )


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


class TestCodeSnippet:
    def test_basic_snippet(self):
        lines = ["line1", "line2", "line3", "line4", "line5"]

        result = code_snippet(lines, 1, 1)
        assert result == ["1| line1"]

        result = code_snippet(lines, 1, 2)
        assert result == ["1| line1", "2| line2"]

        result = code_snippet(lines, 1, 3)
        assert result == ["1| line1", "2| line2", "3| line3"]

    def test_padding(self):
        lines = ["line1", "line2", "line3", "line4", "line5"]

        result = code_snippet(lines, 2, 3, padding_size=1)
        assert result == ["1| line1", "2| line2", "3| line3", "4| line4"]

        result = code_snippet(lines, 2, 3, padding_size=3)
        assert result == ["1| line1", "2| line2", "3| line3", "4| line4", "5| line5"]

    def test_start_line_override(self):
        lines = ["line1", "line2", "line3"]
        result = code_snippet(lines, 1, 2, start_line_override=10)
        assert result == ["10| line1", "11| line2"]

    def test_start_line_override_with_padding(self):
        lines = ["line1", "line2", "line3", "line4", "line5"]
        result = code_snippet(lines, 1, 3, padding_size=1, start_line_override=10)
        assert result == ["10| line1", "11| line2", "12| line3", "13| line4"]

    def test_bad_line_numbers(self):
        lines = ["line1", "line2", "line3"]
        with pytest.raises(
            ValueError, match="start_line and end_line must be greater than 0. They're 1-indexed."
        ):
            code_snippet(lines, 0, 1)
