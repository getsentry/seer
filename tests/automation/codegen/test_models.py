import pytest

from seer.automation.codebase.models import PrFile, PullRequest


class TestPullRequest:
    def test_format_title_and_description_both_present(self):
        """Test formatting when both title and description are present."""
        pr = PullRequest(
            files=[],
            title="Fix critical bug",
            description="This PR fixes a critical bug in the authentication system",
        )

        result = pr.format_title_and_description()

        assert "Pull request title: Fix critical bug" in result
        assert "Pull request description:" in result
        assert "This PR fixes a critical bug in the authentication system" in result
        assert "truncated" not in result

    def test_format_title_and_description_long_description(self):
        """Test formatting when description exceeds max length."""
        long_description = "A" * 9000  # Longer than default max_description_length of 8000
        pr = PullRequest(files=[], title="Fix bug", description=long_description)

        result = pr.format_title_and_description()

        assert "Pull request title: Fix bug" in result
        assert "Pull request description:" in result
        assert "truncated due to excessive length" in result
        assert len(result) < len(long_description)  # Should be truncated

    @pytest.mark.parametrize(
        "title,description,expected_title,expected_description",
        [
            (None, None, False, False),
            ("Test Title", None, True, False),
            (None, "Test Description", False, True),
            ("Test Title", "Test Description", True, True),
        ],
    )
    def test_format_title_and_description_combinations(
        self, title, description, expected_title, expected_description
    ):
        """Test all combinations of title and description being None or present."""
        pr = PullRequest(files=[], title=title, description=description)

        result = pr.format_title_and_description()

        if expected_title:
            assert f"Pull request title: {title}" in result
        else:
            assert "Pull request title:" not in result

        if expected_description:
            assert "Pull request description:" in result
            assert description in result
        else:
            assert "Pull request description:" not in result

    def test_format_diff_empty_files(self):
        """Test formatting diff when no files are present."""
        pr = PullRequest(files=[])

        result = pr.format_diff()

        assert result == "<diff>\n\n\n\n</diff>"

    def test_format_diff_with_files(self):
        """Test formatting diff with multiple files."""
        files = [
            PrFile(
                filename="src/test.py",
                patch="""@@ -1,3 +1,4 @@
 def hello():
+    print("debug")
     return "world"
""",
                status="modified",
                changes=1,
                sha="abc123",
                previous_filename="",
                repo_full_name="test/repo",
            ),
            PrFile(
                filename="src/new_file.py",
                patch="""@@ -0,0 +1,2 @@
+def new_function():
+    pass
""",
                status="added",
                changes=2,
                sha="def456",
                previous_filename="",
                repo_full_name="test/repo",
            ),
        ]

        pr = PullRequest(files=files)

        result = pr.format_diff()

        assert result.startswith("<diff>")
        assert result.endswith("</diff>")
        assert "src/test.py" in result
        assert "src/new_file.py" in result
        assert "Here are the changes made to file" in result

    def test_format_complete_pr(self):
        """Test complete PR formatting including title, description, and diff."""
        files = [
            PrFile(
                filename="README.md",
                patch="""@@ -1,2 +1,3 @@
 # Project
+
 This is a test project.
""",
                status="modified",
                changes=1,
                sha="xyz789",
                previous_filename="",
                repo_full_name="test/repo",
            )
        ]

        pr = PullRequest(
            files=files,
            title="Update README",
            description="Add spacing to README for better formatting",
        )

        result = pr.format()

        # Should contain title and description
        assert "Pull request title: Update README" in result
        assert "Pull request description:" in result
        assert "Add spacing to README for better formatting" in result

        # Should contain diff
        assert "<diff>" in result
        assert "</diff>" in result
        assert "README.md" in result

    def test_format_pr_with_removed_file(self):
        """Test formatting PR with a removed file."""
        files = [
            PrFile(
                filename="old_file.py",
                patch="",  # Removed files might have empty patches
                status="removed",
                changes=0,
                sha="removed123",
                previous_filename="",
                repo_full_name="test/repo",
            )
        ]

        pr = PullRequest(files=files, title="Remove deprecated file", description=None)

        result = pr.format()

        assert "Pull request title: Remove deprecated file" in result
        assert "Pull request description:" not in result
        assert "File old_file.py was removed" in result

    def test_format_pr_with_renamed_file(self):
        """Test formatting PR with a renamed file."""
        files = [
            PrFile(
                filename="new_name.py",
                patch="""@@ -1,1 +1,1 @@
-# Old comment
+# New comment
""",
                status="renamed",
                changes=1,
                sha="rename123",
                previous_filename="old_name.py",
                repo_full_name="test/repo",
            )
        ]

        pr = PullRequest(files=files, title=None, description="Renamed file and updated comment")

        result = pr.format()

        assert "Pull request title:" not in result
        assert "Pull request description:" in result
        assert "Renamed file and updated comment" in result
        assert "File old_name.py was renamed to new_name.py" in result
