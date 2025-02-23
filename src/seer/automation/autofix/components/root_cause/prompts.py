import textwrap
from typing import Optional

from seer.automation.autofix.prompts import (
    format_code_map,
    format_instruction,
    format_repo_names,
    format_summary,
)
from seer.automation.models import Profile
from seer.automation.summarize.issue import IssueSummary


class RootCauseAnalysisPrompts:
    @staticmethod
    def format_system_msg(has_tools: bool = True):
        return textwrap.dedent(
            f"""\
            You are an exceptional AI system that is amazing at researching bugs in codebases.

            {
                "You have tools to search a codebase to gather relevant information. Please use the tools as many times as you want to gather relevant information."
                if has_tools
                else ""
            }

            # Guidelines:
            - Your job is to simply gather all information needed to understand what happened, not to propose fixes.
            - You are not able to search in external libraries. If the error is caused by an external library or the stacktrace only contains frames from external libraries, do not attempt to search in external libraries.
            - At EVERY step of your investigation, you MUST think out loud! Share what you're learning and thinking along the way, EVERY TIME YOU SPEAK.

            It is important that you gather all information needed to understand what happened, from the entry point of the code to the error."""
        )

    @staticmethod
    def format_default_msg(
        event: str,
        repo_names: list[str],
        instruction: Optional[str] = None,
        summary: Optional[IssueSummary] = None,
        code_map: Optional[Profile] = None,
    ):
        return textwrap.dedent(
            """\
            {repo_names_str}
            Given the issue: {summary_str}
            {error_str}

            {code_map_str}

            {instruction_str}
            Gather all information needed to understand what happened.
            At EVERY step of your investigation, you MUST think out loud! Share what you're learning and thinking along the way, EVERY TIME YOU SPEAK."""
        ).format(
            error_str=event,
            repo_names_str=format_repo_names(repo_names),
            instruction_str=format_instruction(instruction),
            summary_str=format_summary(summary),
            code_map_str=format_code_map(code_map),
        )

    @staticmethod
    def root_cause_proposal_msg():
        return textwrap.dedent(
            """\
            Based on all the information you've learned, detail the true root cause of the issue.
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
            As a whole, this timeline should tell the precise story of the root cause of the issue. Starts at the entry point of the code, ends at the error."""
        )
