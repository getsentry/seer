from seer.automation.codebase.utils import potential_frame_match
from seer.automation.models import StacktraceFrame
from seer.automation.utils import encode_decode_base64


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
