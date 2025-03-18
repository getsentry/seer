import textwrap

from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.prompts import format_code_map, format_instruction, format_summary
from seer.automation.models import Profile
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
    ):
        return textwrap.dedent(
            """\
            <goal>Gather all information that may be needed to fix this issue at its root. (don't worry about proposing the fix yet)</goal>

            <available_repos>
            {repos_str}
            </available_repos>

            <issue_details>
            <summary>
            {summary_str}
            </summary>

            <root_cause>
            {root_cause_str}
            </root_cause>

            <raw_issue_details>
            {event_str}
            </raw_issue_details>

            <map_of_relevant_code>
            {code_map_str}
            </map_of_relevant_code>
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
            summary_str=format_summary(summary),
            code_map_str=format_code_map(code_map),
        )

    @staticmethod
    def solution_formatter_msg(root_cause: RootCauseAnalysisItem | str):
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
        ).format(root_cause_str=SolutionPrompts.format_root_cause(root_cause))

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
