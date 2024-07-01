import textwrap
import unittest
from unittest.mock import MagicMock, patch

from seer.automation.autofix.components.snippet_replacement import SnippetReplacementOutput
from seer.automation.autofix.tools import CodeActionTools
from seer.automation.models import FileChange


class TestReplaceSnippetWith(unittest.TestCase):
    @patch(
        "seer.automation.autofix.components.snippet_replacement.SnippetReplacementComponent.invoke"
    )
    def test_replace_snippet_with_success(self, mock_invoke):

        # Setup
        original_snippet = "print('Hello, world!')"
        replacement_snippet = "print('Goodbye, world!')"
        file_path = "test_file.py"
        mock_context = MagicMock()
        contents = textwrap.dedent(
            """\
            def foo():
                print('Hello, world!')
                return True
            """
        )
        mock_context.get_file_contents.return_value = contents

        completion_with_parser = MagicMock()
        code = textwrap.dedent(
            """\
            def foo():
                print('Goodbye, world!')
                return True
            """
        )
        completion_with_parser.return_value = ({"code": code}, MagicMock(), MagicMock())
        mock_invoke.return_value = SnippetReplacementOutput(snippet=code)

        code_action_tools = CodeActionTools(context=mock_context)
        code_action_tools.store_file_change = MagicMock()

        result = code_action_tools.replace_snippet_with(
            file_path, "repo", original_snippet, replacement_snippet, "message"
        )

        # Assert
        self.assertTrue(result)
        self.assertEqual(result, f"success: Resulting code after replacement:\n```\n{code}\n```\n")
        code_action_tools.store_file_change.assert_called_once_with(
            "repo",
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=contents,
                new_snippet=code,
                description="message",
            ),
        )

    @patch(
        "seer.automation.autofix.components.snippet_replacement.SnippetReplacementComponent.invoke"
    )
    def test_replace_snippet_with_chunk_newlines(self, mock_invoke):
        # Setup
        original_snippet = "print('Hello, world!')"
        replacement_snippet = "print('Goodbye, world!')"
        file_path = "test_file.py"
        contents = textwrap.dedent(
            """\

            def foo():
                print('Hello, world!')
                return True


            """
        )
        mock_context = MagicMock()
        mock_context.get_file_contents.return_value = contents

        completion_with_parser = MagicMock()
        code = textwrap.dedent(
            """\
            def foo():
                print('Goodbye, world!')
                return True"""
        )
        completion_with_parser.return_value = ({"code": code}, MagicMock(), MagicMock())
        mock_invoke.return_value = SnippetReplacementOutput(snippet=code)

        code_action_tools = CodeActionTools(context=mock_context)
        code_action_tools.store_file_change = MagicMock()

        result = code_action_tools.replace_snippet_with(
            file_path, "repo", original_snippet, replacement_snippet, "message"
        )

        # Assert
        self.assertTrue(result)
        self.assertEqual(result, f"success: Resulting code after replacement:\n```\n{code}\n```\n")
        code_action_tools.store_file_change.assert_called_once_with(
            "repo",
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=contents.strip("\n") + "\n",
                new_snippet=code + "\n",
                description="message",
            ),
        )

        expected_final_document_content = textwrap.dedent(
            """\

            def foo():
                print('Goodbye, world!')
                return True


            """
        )
        assert (
            code_action_tools.store_file_change.call_args[0][1].apply(contents)
            == expected_final_document_content
        )
