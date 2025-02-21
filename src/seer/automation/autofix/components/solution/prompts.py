import textwrap

from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.prompts import (
    format_code_map,
    format_instruction,
    format_repo_names,
    format_summary,
)
from seer.automation.models import EventDetails, Profile
from seer.automation.summarize.issue import IssueSummary


class SolutionPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional AI system that is amazing at researching bugs in codebases.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets.

            # Guidelines:
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer.

            It is important that you gather all information needed to understand how to fix the issue, from the entry point of the code to the error."""
        )

    @staticmethod
    def format_original_instruction(instruction: str):
        return f"Earlier, the user provided context: {format_instruction(instruction)}"

    @staticmethod
    def format_root_cause(root_cause: RootCauseAnalysisItem | str):
        if isinstance(root_cause, RootCauseAnalysisItem):
            return f"""The steps to reproduce the root cause of the issue have been identified: {RootCausePlanTaskPromptXml.from_root_cause(
                    root_cause
                ).to_prompt_str()}"""
        else:
            return f"The user has provided the following instruction for the fix: {root_cause}"

    @staticmethod
    def format_is_obvious_msg(
        summary: IssueSummary | None,
        event_details: EventDetails,
        root_cause: RootCauseAnalysisItem | str,
        original_instruction: str | None,
    ):
        return (
            textwrap.dedent(
                """\
                Here is an issue in our codebase: {summary_str}

                {event_details}{original_instruction}
                {root_cause_str}

                Does the solution exist ONLY in files you can already see in your context here or do you need to look at other files?"""
            )
            .format(
                summary_str=format_summary(summary),
                event_details=event_details.format_event(),
                root_cause_str=SolutionPrompts.format_root_cause(root_cause),
                original_instruction=(
                    ("\n" + SolutionPrompts.format_original_instruction(original_instruction))
                    if original_instruction
                    else ""
                ),
            )
            .strip()
        )

    @staticmethod
    def format_default_msg(
        *,
        event: str,
        repo_names: list[str],
        root_cause: RootCauseAnalysisItem | str,
        original_instruction: str | None,
        summary: IssueSummary | None,
        code_map: Profile | None,
        has_tools: bool,
    ):
        return textwrap.dedent(
            """\
            {repo_names_str}
            Given the issue: {summary_str}
            {event_str}{original_instruction}
            {root_cause_str}

            {code_map_str}

            GOAL: Gather all information that may be needed to plan a fix for this issue.

            # Guidelines:
            - Your job is to simply gather all information needed. You may not propose code changes yourself.
            {think_tools_instructions}"""
        ).format(
            event_str=event,
            repo_names_str=format_repo_names(repo_names),
            root_cause_str=SolutionPrompts.format_root_cause(root_cause),
            original_instruction=(
                ("\n" + SolutionPrompts.format_original_instruction(original_instruction))
                if original_instruction
                else ""
            ),
            summary_str=format_summary(summary),
            think_tools_instructions=(
                "- EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you."
                if has_tools
                else ""
            ),
            code_map_str=format_code_map(code_map),
        )

    @staticmethod
    def solution_formatter_msg(root_cause: RootCauseAnalysisItem | str):
        return textwrap.dedent(
            """\
            Given the root cause of the issue:
            {root_cause_str}

            Based on the discussed plan, write the steps needed to fix the issue.

            For each step in the plan, include the following:
              - Title: a complete sentence describing what needs to change to fix the issue.
              - Code Snippet and Analysis: an explanation of the code change and the reasoning behind it. All Markdown formatted. (don't write the full code, just tiny snippets at most)
              - Event type: whether this change is about logic in the code, a human interaction, or an external system like a database, API, etc.
              - Is new event: whether this change is the SINGLE MOST important part of the solution.
            As a whole, this sequence of steps should tell the precise plan of how to fix the issue. You can put as few steps as needed.
            """
        ).format(root_cause_str=SolutionPrompts.format_root_cause(root_cause))

    @staticmethod
    def solution_proposal_msg():
        return textwrap.dedent(
            """\
            Based on all the information gathered, provide the most actionable and effective steps to fix the issue. You should propose code changes for each step."""
        )
