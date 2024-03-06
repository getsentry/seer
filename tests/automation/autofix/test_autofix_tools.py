import textwrap
import unittest
from unittest.mock import MagicMock, patch

from seer.automation.autofix.models import FileChange
from seer.automation.autofix.tools import CodeActionTools


class TestReplaceSnippetWith(unittest.TestCase):
    def setUp(self):
        self.mock_context = MagicMock()
        self.code_action_tools = CodeActionTools(context=self.mock_context)

    @patch("seer.automation.agent.client.GptClient")
    def test_replace_snippet_with_success(self, mock_gpt_client):
        # Setup
        original_snippet = "print('Hello, world!')"
        replacement_snippet = "print('Goodbye, world!')"
        file_path = "test_file.py"
        mock_document = MagicMock()
        mock_document.text = textwrap.dedent(
            """\
            def foo():
                print('Hello, world!')
                return True"""
        )
        mock_codebase = MagicMock()
        self.mock_context.get_document_and_codebase.return_value = (mock_codebase, mock_document)

        completion_with_parser = MagicMock()
        code = textwrap.dedent(
            """\
            def foo():
                print('Goodbye, world!')
                return True"""
        )
        completion_with_parser.return_value = ({"code": code}, MagicMock(), MagicMock())
        mock_gpt_client.return_value.completion_with_parser = completion_with_parser

        result = self.code_action_tools.replace_snippet_with(
            file_path, "repo", original_snippet, replacement_snippet, "message"
        )

        # Assert
        self.assertTrue(result)
        self.assertEquals(result, f"success: Resulting code after replacement: ```{code}```")
        mock_codebase.store_file_change.assert_called_once_with(
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=mock_document.text,
                new_snippet=code,
                description="message",
            )
        )
