import textwrap
from typing import Optional

from seer.automation.autofix.components.coding.models import (
    FuzzyDiffChunk,
    PlanStepsPromptXml,
    RootCausePlanTaskPromptXml,
)
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.prompts import format_instruction, format_summary
from seer.automation.models import EventDetails
from seer.automation.summarize.issue import IssueSummary


class CodingPrompts:
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
    def format_extra_root_cause_instruction(instruction: str):
        return f"The user has provided the following instruction for the fix along with the root cause: {format_instruction(instruction)}"

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
    def format_custom_solution(custom_solution: str | None):
        if not custom_solution:
            return ""

        return f"The user has provided the following solution idea: {custom_solution}"

    @staticmethod
    def format_fix_msg(has_tools: bool = True, custom_solution: str | None = None):
        return textwrap.dedent(
            """\
            Break down the task of fixing the issue into steps. Your list of steps should be detailed enough so that following it exactly will lead to a fully complete solution. {custom_solution_str}

            Enclose this plan between <plan_steps> and </plan_steps> tags. Make sure to strictly follow this format and include all necessary details within the tags. Your output must follow the format properly according to the following guidelines:

            {steps_example_str}

            # Guidelines:
            - Each file change must be a separate step and be explicit and clear.
              - You MUST include exact file paths for each step you provide. If you cannot, find the correct path.
            - No placeholders are allowed, the steps must be clear and detailed.
            {use_tools_instructions}
            - The plan must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - Make sure any new files you create don't already exist, if they do, modify the existing file.
            {think_tools_instructions}
            - You also MUST think step-by-step before giving the final answer."""
        ).format(
            steps_example_str=PlanStepsPromptXml.get_example().to_prompt_str(),
            use_tools_instructions=(
                "- Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting the steps."
                if has_tools
                else ""
            ),
            think_tools_instructions=(
                "- EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you."
                if has_tools
                else ""
            ),
            custom_solution_str=CodingPrompts.format_custom_solution(custom_solution),
        )

    @staticmethod
    def format_incorrect_diff_fixer(
        file_path: str, diff_chunks: list[FuzzyDiffChunk], file_content: str
    ):
        return textwrap.dedent(
            """\
            Given the below file content:
            <file path="{file_path}">
            {file_content}
            </file>

            The following diffs were found to be incorrect:
            {diff_chunks}

            Provide the corrected unified diffs inside a <corrected_diffs></corrected_diffs> block:"""
        ).format(
            file_path=file_path,
            file_content=file_content,
            diff_chunks="\n".join([chunk.diff_content for chunk in diff_chunks]),
        )

    @staticmethod
    def format_missing_msg(missing_files: list[str], existing_files: list[str]):
        text = ""

        if missing_files:
            text += f"The following files don't exist: {', '.join(missing_files)}\n"
        if existing_files:
            text += f"The following files already exist: {', '.join(existing_files)}\n"

        return text

    @staticmethod
    def format_is_obvious_msg(
        summary: Optional[IssueSummary],
        event_details: EventDetails,
        root_cause: RootCauseAnalysisItem | str,
        original_instruction: str | None,
        root_cause_extra_instruction: str | None,
        custom_solution: str | None,
    ):
        return (
            textwrap.dedent(
                """\
                Here is an issue in our codebase: {summary_str}

                {event_details}{original_instruction}
                {root_cause_str}{root_cause_extra_instruction}

                {custom_solution_str}
                Does the code change exist ONLY in files you can already see in your context here or do you need to look at other files?"""
            )
            .format(
                summary_str=format_summary(summary),
                event_details=event_details.format_event(),
                root_cause_str=CodingPrompts.format_root_cause(root_cause),
                original_instruction=(
                    ("\n" + CodingPrompts.format_original_instruction(original_instruction))
                    if original_instruction
                    else ""
                ),
                root_cause_extra_instruction=(
                    (
                        "\n"
                        + CodingPrompts.format_extra_root_cause_instruction(
                            root_cause_extra_instruction
                        )
                    )
                    if root_cause_extra_instruction
                    else ""
                ),
                custom_solution_str=CodingPrompts.format_custom_solution(custom_solution),
            )
            .strip()
        )
