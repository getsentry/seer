import textwrap
import unittest

from seer.automation.autofix.utils import (
    VALID_RANDOM_SUFFIX_CHARS,
    find_original_snippet,
    generate_random_string,
    sanitize_branch_name,
)


class TestFindOriginalSnippet(unittest.TestCase):
    def test_find_original_snippet_exact_match(self):
        snippet = "def example_function():\n    pass"
        file_contents = textwrap.dedent(
            """\
            def unrelated_function():
                print("Hello, world!")

            def example_function():
                pass

            def another_function():
                print("Goodbye, world!")
            """
        )
        self.assertEqual(find_original_snippet(snippet, file_contents), (snippet, 3, 5))

    def test_find_original_snippet_with_whitespace_variation(self):
        snippet = "def example_function():\n    pass"
        file_contents = textwrap.dedent(
            """\
            def example_function():
                    pass
            """
        )
        self.assertIsNotNone(find_original_snippet(snippet, file_contents))

    def test_find_original_snippet_not_found(self):
        snippet = "def missing_function():\n    pass"
        file_contents = textwrap.dedent(
            """\
            def example_function():
                pass
            """
        )
        self.assertIsNone(find_original_snippet(snippet, file_contents))

    def test_find_original_snippet_multiple_similar_pieces(self):
        snippet = "def example_function():\n    print('Hello')"
        file_contents = textwrap.dedent(
            """\
            def example_function():
                print('Hi')

            def unrelated_function():
                pass

            def example_function():
                print('Hello')

            def another_example_function():
                print('Hello')
            """
        )
        expected_result = ("def example_function():\n    print('Hello')", 6, 8)
        self.assertEqual(
            find_original_snippet(snippet, file_contents, threshold=0.95), expected_result
        )

    def test_find_original_snippet_fuzzy_match(self):
        snippet = "def example_function():\n    print('Hello, world!')"
        file_contents = textwrap.dedent(
            """\
            def example_function():
                print('Hello world!')

            def unrelated_function():
                pass
            """
        )  # missing comma
        expected_result = (
            "def example_function():\n    print('Hello world!')",
            0,
            2,
        )
        self.assertEqual(
            find_original_snippet(snippet, file_contents, threshold=0.75), expected_result
        )

    def test_find_original_snippet_empty_snippet(self):
        snippet = ""
        file_contents = textwrap.dedent(
            """\
            def example_function():
                pass
            """
        )
        self.assertIsNone(find_original_snippet(snippet, file_contents))


class TestSanitizeBranchName(unittest.TestCase):
    def test_basic_sanitization(self):
        """Test that spaces and underscores are converted to hyphens and text is lowercased."""
        title = "Hello World_Example"
        expected = "hello-world-example"
        self.assertEqual(sanitize_branch_name(title), expected)

    def test_removes_invalid_characters(self):
        """Test that characters not in VALID_BRANCH_NAME_CHARS are removed."""
        title = "feature!@#$%^&*(-)+=;:'\"<>,.?branch"
        expected = "feature-branch"
        self.assertEqual(sanitize_branch_name(title), expected)

    def test_strips_trailing_slashes(self):
        """Test the new functionality that strips trailing slashes."""
        title = "feature/branch/"
        expected = "feature/branch"
        self.assertEqual(sanitize_branch_name(title), expected)

        # Test with multiple trailing slashes
        title = "feature/branch///"
        expected = "feature/branch"
        self.assertEqual(sanitize_branch_name(title), expected)

    def test_does_not_strip_middle_slashes(self):
        """Test that slashes in the middle of the branch name are preserved."""
        title = "feature/branch/name"
        expected = "feature/branch/name"
        self.assertEqual(sanitize_branch_name(title), expected)


class TestGenerateRandomString(unittest.TestCase):
    def test_output_length(self):
        """Test that the output string has the expected length."""
        length = 8
        result = generate_random_string(length)
        self.assertEqual(len(result), length)

        # Test default length
        result = generate_random_string()
        self.assertEqual(len(result), 6)

    def test_characters_used(self):
        """Test that only characters from VALID_RANDOM_SUFFIX_CHARS are used."""
        # Test with a longer string to have a higher chance of detecting issues
        result = generate_random_string(100)
        for char in result:
            self.assertIn(char, VALID_RANDOM_SUFFIX_CHARS)

    def test_no_invalid_chars(self):
        """Test that slashes and dashes are not included in the generated string."""
        result = generate_random_string(1000)  # Generate a long string for better testing
        self.assertNotIn("-", result)
        self.assertNotIn("/", result)

    def test_randomness(self):
        """Test that two calls produce different results."""
        result1 = generate_random_string()
        result2 = generate_random_string()
        self.assertNotEqual(result1, result2)
