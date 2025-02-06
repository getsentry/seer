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
    def format_system_msg(has_tools: bool):
        if has_tools:
            return textwrap.dedent(
                """\
                You are an exceptional principal engineer that is amazing at finding and fixing issues in codebases.

                You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets.

                # Guidelines:
                - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
                - You also MUST think step-by-step before giving the final answer."""
            )
        else:
            return textwrap.dedent(
                """\
                You are an exceptional principal engineer that is amazing at finding and fixing issues in codebases.

                # Guidelines:
                - You also MUST think step-by-step before giving the final answer."""
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

            # Your goal:
            Provide the most actionable and effective steps to fix the issue.

            Since you are an exceptional principal engineer, your solution should not just add logs or throw more errors, but should meaningfully fix the issue. Your list of steps to fix the problem should be detailed enough so that following it exactly will lead to a fully complete solution.

            When ready with your final answer, detail the precise plan to fix the issue.

            # Guidelines:
            - No placeholders are allowed, the fix must be clear and detailed.
            {use_tools_instructions}
            - The fix must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - If the issue occurs in multiple places or files, make sure to provide a fix for each occurrence, no matter how many there are.
            - In your suggested fixes, whenever you are providing code, provide explicit diffs to show the exact changes that need to be made.
            - You do not need to make changes in test files, someone else will do that.
            {ask_questions_instructions}
            {search_google_instructions}
            {think_tools_instructions}
            - You also MUST think step-by-step before giving the final answer."""
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
            use_tools_instructions=(
                "- Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting your suggested fix."
                if has_tools
                else ""
            ),
            think_tools_instructions=(
                "- EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you."
                if has_tools
                else ""
            ),
            ask_questions_instructions=(
                "- At any point, please feel free to ask your teammates any specific questions that would help you in your analysis that you cannot answer from analyzing the code."
                if has_tools
                else ""
            ),
            search_google_instructions=(
                "- At any point, please feel free to Google for information that would help you in your analysis, using the tool provided."
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

            Rewrite this timeline exactly as is, but remove/replace/add items, so that the final timeline illustrates the planned solution to this issue: how the code SHOULD operate step by step.

            For each event:
              - Title: a complete sentence describing what happened and why it matters to the root cause of the issue. (a summary of the description)
              - Code Snippet and Analysis: any extra analysis needed and a small relevant code snippet if this is an important event. All Markdown formatted.
              - Event type: logic in the code, a human interaction, or an external system like a database, API, etc.
              - Is new event: whether this item in the timeline is a proposed code change, part of the solution.
            As a whole, this timeline should tell the precise story of how the code should work to fix the issue. Starts at the entry point of the code, and end at the ideal outcome.
            """
        ).format(root_cause_str=SolutionPrompts.format_root_cause(root_cause))
