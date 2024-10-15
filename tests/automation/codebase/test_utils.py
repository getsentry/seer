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


class TestEncodeDecodeBase64:
    def test_encode_base64(self):
        input_string = "Hello, World!"
        encoded = encode_decode_base64(input_string, operation="encode")
        assert encoded == "SGVsbG8sIFdvcmxkIQ=="

    def test_decode_base64(self):
        input_string = "SGVsbG8sIFdvcmxkIQ=="
        decoded = encode_decode_base64(input_string, operation="decode")
        assert decoded == "Hello, World!"

    def test_encode_decode_roundtrip(self):
        original = "Test string with special chars: !@#$%^&*()"
        encoded = encode_decode_base64(original, operation="encode")
        decoded = encode_decode_base64(encoded, operation="decode")
        assert decoded == original

    def test_encode_empty_string(self):
        assert encode_decode_base64("", operation="encode") == ""

    def test_decode_empty_string(self):
        assert encode_decode_base64("", operation="decode") == ""

    def test_invalid_operation(self):
        with pytest.raises(ValueError) as exc_info:
            encode_decode_base64("test", operation="invalid")
        assert str(exc_info.value) == "Invalid operation. Choose 'encode' or 'decode'."

    def test_decode_invalid_base64(self):
        with pytest.raises(Exception):
            encode_decode_base64("invalid base64!", operation="decode")
