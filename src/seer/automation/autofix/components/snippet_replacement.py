import textwrap

from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.automation.utils import extract_text_inside_tags, get_autofix_client_and_agent


class SnippetReplacementRequest(BaseComponentRequest):
    reference_snippet: str
    replacement_snippet: str
    chunk: str
    commit_message: str


class SnippetReplacementOutput(BaseComponentOutput):
    snippet: str


class SnippetReplacementPrompts:
    @staticmethod
    def format_default_msg(
        reference_snippet: str, replacement_snippet: str, chunk: str, commit_message: str
    ):
        return textwrap.dedent(
            """\
            Replace the following snippet:

            <snippet>
            {reference_snippet}
            </snippet>

            with the following snippet:
            <snippet>
            {replacement_snippet}
            </snippet>

            in the below chunk of code:
            <chunk>
            {chunk}
            </chunk>

            The intent of this change is
            <description>
            {commit_message}
            </description>

            <requirements>
            - Make sure you fix any errors in the code and ensure it is working as expected to the intent of the change.
            - Do not make extraneous changes to the code or whitespace that are not related to the intent of the change.
            - Make sure you maintain code indentation and formatting as per the original code chunk in your output.
            - You MUST return the code result inside a <code></code> tag.
            </requirements>"""
        ).format(
            reference_snippet=reference_snippet,
            replacement_snippet=replacement_snippet,
            chunk=chunk,
            commit_message=commit_message,
        )


class SnippetReplacementComponent(
    BaseComponent[SnippetReplacementRequest, SnippetReplacementOutput]
):
    context: AutofixContext

    def _parser(self, text: str | None):
        return extract_text_inside_tags(text, "code", strip_newlines=True) if text else None

    def invoke(self, request: SnippetReplacementRequest) -> SnippetReplacementOutput | None:
        prompt = SnippetReplacementPrompts.format_default_msg(
            reference_snippet=request.reference_snippet,
            replacement_snippet=request.replacement_snippet,
            chunk=request.chunk,
            commit_message=request.commit_message,
        )

        data, message, usage = get_autofix_client_and_agent()[0]().completion_with_parser(
            [Message(role="user", content=prompt)], parser=self._parser
        )

        with self.context.state.update() as cur:
            cur.usage += usage

        if data is None:
            return None

        return SnippetReplacementOutput(snippet=data)
