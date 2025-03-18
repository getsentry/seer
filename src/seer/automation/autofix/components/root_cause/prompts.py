import textwrap
from typing import Optional

from seer.automation.autofix.prompts import format_code_map, format_instruction, format_summary
from seer.automation.models import Profile
from seer.automation.summarize.issue import IssueSummary


class RootCauseAnalysisPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional AI system that is amazing at researching bugs in codebases.

            You have tools to search a codebase to gather relevant information. Please use the tools as many times as you want to gather relevant information.

            # Guidelines:
            - Your job is to simply gather all information needed to understand what happened, not to propose fixes.
            - You are not able to search in external libraries. If the error is caused by an external library or the stacktrace only contains frames from external libraries, do not attempt to search in external libraries.

            It is important that you gather all information needed to understand what happened, from the entry point of the code to the error."""
        )

    @staticmethod
    def format_default_msg(
        event: str,
        repos_str: str,
        instruction: Optional[str] = None,
        summary: Optional[IssueSummary] = None,
        code_map: Optional[Profile] = None,
    ):
        return textwrap.dedent(
            """\
            <goal>{explore_msg}</goal>
            <available_repos>
            {repos_str}
            </available_repos>

            <issue_details>
            Given the issue: {summary_str}
            {error_str}
            {code_map_str}
            {instruction_str}
            </issue_details>"""
        ).format(
            explore_msg=(
                "Gather all information needed to understand what happened, from the entry point of the code to the error."
            ),
            error_str=event,
            repos_str=repos_str,
            instruction_str=format_instruction(instruction),
            summary_str=format_summary(summary),
            code_map_str=format_code_map(code_map),
        )

    @staticmethod
    def root_cause_proposal_msg():
        return textwrap.dedent(
            """\
            <goal>Based on all the information you've learned, detail the true root cause of the issue.</goal>
            """
        )

    @staticmethod
    def root_cause_formatter_msg():
        return textwrap.dedent(
            """\
            Write a reproduction timeline to illustrate how exactly the issue occurred and why.

            For each event:
              - Title: a complete sentence describing what happened and why it matters to the root cause of the issue. (a summary of the description)
              - Code Snippet and Analysis: any extra analysis needed and a small relevant code snippet if this is an important event. All Markdown formatted.
              - Event type: logic in the code, a human interaction, or an external system like a database, API, etc.
              - Is most important event: whether this event is the MOST important and insightfulone in the timeline to pay attention to.
            As a whole, this timeline should tell the precise story of the root cause of the issue. Starts at the entry point of the code, ends at the error.

            Then, provide a concise summary of the root cause of the issue. This summary must be less than 30 words and must be an information-dense single summary and must not contain filler words such as "The application..." or "The issue...".
              - Use a "matter of fact" tone, such as "The `process_task` function did not check if the task was already processed, due to `foo` being `bar`."."""
        )
