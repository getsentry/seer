import textwrap
import unittest

from seer.automation.autofix.utils import find_original_snippet


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
