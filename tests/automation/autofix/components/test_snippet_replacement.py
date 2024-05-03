import textwrap
import unittest
from unittest.mock import patch

from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.components.snippet_replacement import (
    SnippetReplacementComponent,
    SnippetReplacementRequest,
)


class TestSnippetReplacementComponent(unittest.TestCase):
    @patch("seer.automation.autofix.components.snippet_replacement.AutofixContext")
    @patch("seer.automation.autofix.components.snippet_replacement.GptClient.completion")
    def test_simple(self, mock_completion_with_parser, mock_autofix_context):
        component = SnippetReplacementComponent(mock_autofix_context.return_value)

        mock_completion_with_parser.return_value = (
            Message(
                role="assistant",
                content=textwrap.dedent(
                    """\
                    <code>
                    function foo() {
                        const y = 0;
                    }
                    </code>"""
                ),
            ),
            Usage(),
        )

        output = component.invoke(
            SnippetReplacementRequest(
                reference_snippet="const x = 0",
                replacement_snippet="const y = 0",
                chunk=textwrap.dedent(
                    """\
                    function foo() {
                        const x = 0;
                    }"""
                ),
                commit_message="Message",
            )
        )

        self.assertIsNotNone(output)
        if output is not None:
            self.assertEqual(
                output.snippet,
                textwrap.dedent(
                    """\
                    function foo() {
                        const y = 0;
                    }"""
                ),
            )

    @patch("seer.automation.autofix.components.snippet_replacement.AutofixContext")
    @patch("seer.automation.autofix.components.snippet_replacement.GptClient.completion")
    def test_with_extra_newlines(self, mock_completion_with_parser, mock_autofix_context):
        component = SnippetReplacementComponent(mock_autofix_context.return_value)

        mock_completion_with_parser.return_value = (
            Message(
                role="assistant",
                content=textwrap.dedent(
                    """\
                    <code>

                    function foo() {
                        const y = 0;
                    }


                    </code>"""
                ),
            ),
            Usage(),
        )

        output = component.invoke(
            SnippetReplacementRequest(
                reference_snippet="const x = 0",
                replacement_snippet="const y = 0",
                chunk=textwrap.dedent(
                    """\
                    function foo() {
                        const x = 0;
                    }"""
                ),
                commit_message="Message",
            )
        )

        self.assertIsNotNone(output)
        if output is not None:
            self.assertEqual(
                output.snippet,
                textwrap.dedent(
                    """\
                    function foo() {
                        const y = 0;
                    }"""
                ),
            )
