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
        self.assertEqual(find_original_snippet(snippet, file_contents), snippet)

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

    def test_find_original_snippet_with_ellipsis(self):
        snippet = "def example_function():\n    ...\n    pass"
        file_contents = textwrap.dedent(
            """\
            def example_function():
                print("Doing something")
                pass
            """
        )
        self.assertIsNotNone(find_original_snippet(snippet, file_contents))

    def test_find_original_snippet_with_multiline_ellipsis(self):
        snippet = "def example_function():\n    ...\n    pass\n    # ...\n    pass"
        file_contents = textwrap.dedent(
            """\
            def example_function():
                print("Doing something")
                print("Doing something else")
                pass

            def another_function():
                print("Goodbye, world!")
                pass
            """
        )
        self.assertIsNotNone(find_original_snippet(snippet, file_contents))
