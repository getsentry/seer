import textwrap

from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.prompts import (
    format_code_map,
    format_instruction,
    format_summary,
    format_trace_tree,
)
from seer.automation.models import Profile, TraceTree
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
            return RootCausePlanTaskPromptXml.from_root_cause(root_cause).to_prompt_str()
        else:
            return root_cause

    @staticmethod
    def format_default_msg(
        *,
        event: str,
        repos_str: str,
        root_cause: RootCauseAnalysisItem | str,
        original_instruction: str | None,
        summary: IssueSummary | None,
        code_map: Profile | None,
        trace_tree: TraceTree | None,
    ):
        return textwrap.dedent(
            """\
            <goal>Gather all information that may be needed to understand how to fix this issue at its root in the way the developer would most likely want to fix it. (don't worry about proposing the fix itself yet)</goal>

            <available_repos>
            {repos_str}
            </available_repos>

            <issue_details>
            {summary_str}

            <root_cause>
            {root_cause_str}
            </root_cause>

            <raw_issue_details>
            {event_str}
            </raw_issue_details>

            {code_map_str}
            {trace_tree_str}
            </issue_details>
            """
        ).format(
            event_str=event,
            repos_str=repos_str,
            root_cause_str=SolutionPrompts.format_root_cause(root_cause),
            original_instruction=(
                ("\n" + SolutionPrompts.format_original_instruction(original_instruction))
                if original_instruction
                else ""
            ),
            summary_str=f"<summary>{format_summary(summary)}</summary>" if summary else "",
            code_map_str=(
                f"<map_of_relevant_code>{format_code_map(code_map)}</map_of_relevant_code>"
                if code_map
                else ""
            ),
            trace_tree_str=f"<trace>{format_trace_tree(trace_tree)}</trace>" if trace_tree else "",
        )

    @staticmethod
    def solution_formatter_msg():
        return textwrap.dedent(
            """\
            Format the discussed plan exactly into a list of steps in the plan to fix the issue. Exclude steps that are not part of the fix, such as adding tests and logs.

            For each item in the plan (where one item is one step to fix the issue):
              - Title: a complete sentence describing what needs to change to fix the issue.
              - Code Snippet and Analysis: A snippet of the code change and an explanation of the code change and the reasoning behind it. All Markdown formatted. (don't write the full code, just tiny snippets at most)
              - Is most important: whether this change is the SINGLE MOST important part of the solution.
            As a whole, this sequence of steps should tell the precise plan of how to fix the issue. You can put as few or as many steps as needed.

            Then, provide a concise summary of the solution. This summary must be less than 30 words and must be an information-dense single summary and must not contain filler words such as "The application..." or "The fix...".
              - Use a "matter of fact" tone, such as "Add correct validation of `foo` to the `process_task` function."."""
        )

    @staticmethod
    def solution_proposal_msg():
        return textwrap.dedent(
            """\
            <goal>Based on all the information you've learned, outline the most actionable and effective steps to fix the issue.</goal>

            <guidelines>
            Your solution should be whatever is the BEST, most CORRECT solution, whether it's a one-line change or a bigger refactor.
            </guidelines>
            """
        )
