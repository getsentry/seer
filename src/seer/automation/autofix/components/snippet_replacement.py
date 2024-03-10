import textwrap

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest


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

            Make sure you fix any errors in the code and ensure it is working as expected to the intent of the change.
            Do not make extraneous changes to the code or whitespace that are not related to the intent of the change.

            You MUST return the code result under the "code": key in the response JSON object."""
        ).format(
            reference_snippet=reference_snippet,
            replacement_snippet=replacement_snippet,
            chunk=chunk,
            commit_message=commit_message,
        )


class SnippetReplacementComponent(BaseComponent):
    def invoke(self, request: SnippetReplacementRequest) -> SnippetReplacementOutput | None:
        prompt = SnippetReplacementPrompts.format_default_msg(
            reference_snippet=request.reference_snippet,
            replacement_snippet=request.replacement_snippet,
            chunk=request.chunk,
            commit_message=request.commit_message,
        )

        data, message, usage = GptClient().json_completion(
            [Message(role="user", content=prompt)],
        )

        if data is None:
            return None

        return SnippetReplacementOutput(snippet=data["code"])
