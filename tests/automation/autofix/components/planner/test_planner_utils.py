import pytest

from seer.automation.autofix.components.planner.models import FuzzyDiffChunk
from seer.automation.autofix.components.planner.utils import extract_diff_chunks


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
