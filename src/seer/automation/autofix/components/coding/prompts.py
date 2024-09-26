import textwrap
from typing import Optional

from seer.automation.autofix.components.coding.models import PlanStepsPromptXml
from seer.automation.autofix.prompts import format_instruction, format_repo_names, format_summary
from seer.automation.summarize.issue import IssueSummary


class CodingPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at finding and fixing issues in codebases.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets.

            # Guidelines:
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        )

    @staticmethod
    def format_fix_discovery_msg(
        event: str,
        task_str: str,
        repo_names: list[str],
        instruction: str | None,
        summary: Optional[IssueSummary] = None,
    ):
        return textwrap.dedent(
            """\
            {repo_names_str}
            Given the issue: {summary_str}
            {event_str}

            {instruction}
            The root cause of the issue has been identified and context about the issue has been provided:
            {task_str}

            # Your goal:
            Provide the most actionable and effective steps to fix the issue.

            Since you are an exceptional principal engineer, your solution should not just add logs or throw more errors, but should meaningfully fix the issue. Your list of steps to fix the problem should be detailed enough so that following it exactly will lead to a fully complete solution.

            When ready with your final answer, detail the precise plan to fix the issue.

            # Guidelines:
            - No placeholders are allowed, the fix must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting your suggested fix.
            - The fix must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - If the issue occurs in multiple places or files, make sure to provide a fix for each occurrence, no matter how many there are.
            - In your suggested fixes, whenever you are providing code, provide explicit diffs to show the exact changes that need to be made.
            - You do not need to make changes in test files, someone else will do that.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(
            event_str=event,
            task_str=task_str,
            repo_names_str=format_repo_names(repo_names),
            instruction=format_instruction(instruction),
            summary_str=format_summary(summary),
        )

    @staticmethod
    def format_fix_msg():
        return textwrap.dedent(
            """\
            Break down the task of fixing the issue into steps. Your list of steps should be detailed enough so that following it exactly will lead to a fully complete solution.

            When ready with your final answer, detail the precise plan to accomplish the task wrapped with a <plan_steps></plan_steps> block. Your output must follow the format properly according to the following guidelines:

            {steps_example_str}

            # Guidelines:
            - Each file change must be a separate step and be explicit and clear.
              - You MUST include exact file paths for each step you provide. If you cannot, find the correct path.
            - No placeholders are allowed, the steps must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting the steps.
            - The plan must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - Make sure any new files you create don't already exist, if they do, modify the existing file.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(steps_example_str=PlanStepsPromptXml.get_example().to_prompt_str())
