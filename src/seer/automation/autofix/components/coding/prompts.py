import textwrap
from typing import Literal

from seer.automation.autofix.components.coding.models import (
    CodeChangesPromptXml,
    RootCausePlanTaskPromptXml,
)
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.components.solution.models import SolutionTimelineEvent
from seer.automation.autofix.prompts import format_instruction
from seer.automation.models import EventDetails


class CodingPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at finding and fixing issues in codebases.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets."""
        )

    @staticmethod
    def format_extra_root_cause_instruction(instruction: str):
        return f"The user has provided the following instruction for the fix along with the root cause: {format_instruction(instruction)}"

    @staticmethod
    def format_original_instruction(instruction: str):
        return f"Earlier, the user provided context: {format_instruction(instruction)}"

    @staticmethod
    def format_root_cause(root_cause: RootCauseAnalysisItem | str | None):
        if root_cause is None:
            return ""

        if isinstance(root_cause, RootCauseAnalysisItem):
            return f"""The steps to reproduce the root cause of the issue have been identified: {RootCausePlanTaskPromptXml.from_root_cause(
                    root_cause
                ).to_prompt_str()}"""
        else:
            return f"The user has provided the following instruction for the fix: {root_cause}"

    @staticmethod
    def format_custom_solution(custom_solution: str | None):
        if not custom_solution:
            return "No plan provided."

        return custom_solution

    @staticmethod
    def format_auto_solution(auto_solution: list[SolutionTimelineEvent] | None):
        if not auto_solution:
            return "No plan provided."

        solution_str = ""
        solution_str = "\n".join(
            f"<step_{i+1}>\n{event.title}\n{event.code_snippet_and_analysis}\n</step_{i+1}>"
            for i, event in enumerate(auto_solution)
        )
        return solution_str

    @staticmethod
    def format_fix_msg(
        custom_solution: str | None = None,
        auto_solution: list[SolutionTimelineEvent] | None = None,
        mode: Literal["all", "fix", "test"] = "fix",
        event_details: EventDetails | None = None,
        root_cause: RootCauseAnalysisItem | str | None = None,
    ):
        return textwrap.dedent(
            """\
            <goal>Break down the task of {mode_str} into a list of code changes to make. {filter_str}</goal>

            <output_format>
            Enclose this plan between <code_changes> and </code_changes> tags. Your output must follow the format properly according to the following guidelines:

            {steps_example_str}
            </output_format>

            <solution_plan>
            {solution_str}
            </solution_plan>

            <root_cause_of_issue>
            {root_cause_str}
            </root_cause_of_issue>

            <raw_issue_details>
            {event_details_str}
            </raw_issue_details>
            """
        ).format(
            mode_str=(
                "fixing the issue"
                if mode == "fix"
                else (
                    "writing a unit test to reproduce the issue and assert the planned solution (following test-driven development)"
                    if mode == "test"
                    else "writing a unit test to reproduce the issue and assert the planned solution (following test-driven development) and then fixing the issue"
                )
            ),
            filter_str=(
                "Use the planned solution to inform the test, but do NOT implement the solution. Only write the test."
                if mode == "test"
                else "Your list of steps should be detailed enough so that following it exactly will lead to a fully complete solution."
            ),
            steps_example_str=CodeChangesPromptXml.get_example().to_prompt_str(),
            solution_str=(
                CodingPrompts.format_custom_solution(custom_solution)
                if custom_solution
                else CodingPrompts.format_auto_solution(auto_solution)
            ),
            root_cause_str=CodingPrompts.format_root_cause(root_cause),
            event_details_str=(event_details.format_event() if event_details else ""),
        )

    @staticmethod
    def format_missing_msg(
        missing_files: list[str], existing_files: list[str], correct_paths: list[str]
    ):
        text = ""

        if correct_paths:
            text += (
                f"The following code changes are formatted correctly: {', '.join(correct_paths)}\n"
            )
            text += "But..."
        if missing_files:
            text += f"The following files don't exist, yet you are trying to modify them: {', '.join(missing_files)}\n"
        if existing_files:
            text += f"The following files already exist, yet you are trying to create them: {', '.join(existing_files)}\n"

        text += "\nPlease fix the above issues by correcting the file paths or correcting the type (file_create, file_change, or file_delete) and output your answer in the correct format again. Re-write your WHOLE answer, including the already-correct changes."

        return text

    @staticmethod
    def format_xml_format_fix_msg():
        example = CodeChangesPromptXml.get_example().to_prompt_str()
        return textwrap.dedent(
            """\
            Your previous response had invalid XML formatting. Please provide your response again with valid XML tag, fields, and attributes. Again, here is the example of the correct format:\n{example}"""
        ).format(example=example)

    @staticmethod
    def format_is_obvious_msg(
        root_cause: RootCauseAnalysisItem | str,
        original_instruction: str | None,
        root_cause_extra_instruction: str | None,
        custom_solution: str | None,
        auto_solution: list[SolutionTimelineEvent] | None,
        mode: Literal["all", "fix", "test"] = "fix",
    ):
        return (
            textwrap.dedent(
                """\
                Here is an issue in our codebase:

                {original_instruction}
                {root_cause_str}{root_cause_extra_instruction}

                <solution_plan>
                {solution_str}
                </solution_plan>

                Does the code change needed for {mode_str} exist ONLY in files you can already see in your context here or do you need to look at other files?"""
            )
            .format(
                mode_str=(
                    "fixing the issue"
                    if mode == "fix"
                    else (
                        "writing a unit test to reproduce the issue"
                        if mode == "test"
                        else "writing a unit test and fixing the issue"
                    )
                ),
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
                solution_str=(
                    CodingPrompts.format_custom_solution(custom_solution)
                    if custom_solution
                    else CodingPrompts.format_auto_solution(auto_solution)
                ),
            )
            .strip()
        )
