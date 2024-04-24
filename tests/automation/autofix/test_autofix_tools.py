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
        mock_context = MagicMock()
        code_action_tools = CodeActionTools(context=mock_context)

        # Setup
        original_snippet = "print('Hello, world!')"
        replacement_snippet = "print('Goodbye, world!')"
        file_path = "test_file.py"
        mock_document = MagicMock()
        mock_document.text = textwrap.dedent(
            """\
            def foo():
                print('Hello, world!')
                return True
            """
        )
        mock_codebase = MagicMock()
        mock_context.get_document_and_codebase.return_value = (mock_codebase, mock_document)

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

        result = code_action_tools.replace_snippet_with(
            file_path, "repo", original_snippet, replacement_snippet, "message"
        )

        # Assert
        self.assertTrue(result)
        self.assertEqual(result, f"success: Resulting code after replacement:\n```\n{code}\n```\n")
        mock_codebase.store_file_change.assert_called_once_with(
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=mock_document.text,
                new_snippet=code,
                description="message",
            )
        )

    @patch(
        "seer.automation.autofix.components.snippet_replacement.SnippetReplacementComponent.invoke"
    )
    def test_replace_snippet_with_chunk_newlines(self, mock_invoke):
        mock_context = MagicMock()
        code_action_tools = CodeActionTools(context=mock_context)

        # Setup
        original_snippet = "print('Hello, world!')"
        replacement_snippet = "print('Goodbye, world!')"
        file_path = "test_file.py"
        mock_document = MagicMock()
        mock_document.text = textwrap.dedent(
            """\

            def foo():
                print('Hello, world!')
                return True


            """
        )
        mock_codebase = MagicMock()
        mock_context.get_document_and_codebase.return_value = (mock_codebase, mock_document)

        completion_with_parser = MagicMock()
        code = textwrap.dedent(
            """\
            def foo():
                print('Goodbye, world!')
                return True"""
        )
        completion_with_parser.return_value = ({"code": code}, MagicMock(), MagicMock())
        mock_invoke.return_value = SnippetReplacementOutput(snippet=code)

        result = code_action_tools.replace_snippet_with(
            file_path, "repo", original_snippet, replacement_snippet, "message"
        )

        # Assert
        self.assertTrue(result)
        self.assertEqual(result, f"success: Resulting code after replacement:\n```\n{code}\n```\n")
        mock_codebase.store_file_change.assert_called_once_with(
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=mock_document.text.strip("\n") + "\n",
                new_snippet=code + "\n",
                description="message",
            )
        )

        expected_final_document_content = textwrap.dedent(
            """\

            def foo():
                print('Goodbye, world!')
                return True


            """
        )
        assert (
            mock_codebase.store_file_change.call_args[0][0].apply(mock_document.text)
            == expected_final_document_content
        )
