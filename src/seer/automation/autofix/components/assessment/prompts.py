import textwrap
from typing import Optional

from seer.automation.autofix.models import ExceptionDetails
from seer.automation.autofix.prompts import format_additional_context, format_exceptions


class ProblemDiscoveryPrompts:
    @staticmethod
    def format_default_msg(
        event_title: str,
        exceptions: list[ExceptionDetails],
        additional_context: Optional[str] = None,
    ):
        return textwrap.dedent(
            """\
            Assess the issue:
            <issue>
            <issue_title>
            {issue_title}
            </issue_title>
            <exceptions>
            {exception_strs}
            </exceptions>
            </issue>
            There could be multiple exceptions in an issue, and the last one is the most relevant.

            {additional_context_str}
            Assess the above issue."""
        ).format(
            issue_title=event_title,
            exception_strs=format_exceptions(exceptions),
            additional_context_str=format_additional_context(additional_context),
        )

    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is tasked with finding how actionable the problem is given an error message and stack trace. Think step-by-step before outputting your answer.

            An actionable problem is one where the cause is apparent within the error message and stack trace and would easily be fixed with a simple, straightforward code change. A non-actionable problem is one where the cause is not apparent and would require some investigation to fix and more than a simple, straightforward code change.
            The actionability score should also take into account whether this error should be fixed immediately or can be fixed at a later time.

            - Output a description of the problem and whether it is actionable in JSON format.
            - Provide a description of the problem in the "description" field.
            - Provide a "reasoning" field with your reasoning for why the problem is actionable or not.
            - Inside the "actionability_score" field, output a float score from 0-1.0 if the error message and stack trace is actionable and would be fixed with a simple, straightforward code change.
            - Example format provided below:
            <example_output>
            {
                "reasoning": "This should be actionable because X Y Z",
                "description": "The function here is not working because X Y Z",
                "actionability_score": 0.8
            }
            </example_output>"""
        )
