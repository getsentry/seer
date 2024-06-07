import textwrap

from seer.automation.autofix.components.planner.models import PlanStepsPromptXml
from seer.automation.autofix.prompts import format_exceptions, format_instruction
from seer.automation.models import ExceptionDetails


class PlanningPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that solves problems with the best plans.

            You are giving tasks to a coding agent that will perform code changes based on your instructions. The tasks must be clear and detailed enough that the coding agent can perform the task without any additional information.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets.

            Your output must use the below format and use the types of steps provided:
            {steps_example_str}

            Guidelines:
            - Each code change must be a separate step and be explicit and clear.
            - No placeholders are allowed, the steps must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting the steps.
            - Use the <thoughts> tag to think step by step before you return your result with a <plan_steps> tag."""
        ).format(steps_example_str=PlanStepsPromptXml.get_example().to_prompt_str())

    @staticmethod
    def format_default_msg(
        err_msg: str, exceptions: list[ExceptionDetails], task_str: str, instruction: str | None
    ):
        return textwrap.dedent(
            """\
            Given the issue:
            <issue>
            <error_message>
            {err_msg}
            </error_message>
            {exceptions_str}
            </issue>

            You have to break the below task into steps:
            <task>
            {task_str}
            </task>

            Think step-by-step inside the <thoughts> tag then output a concise and simple list of steps to perform in the output format provided in the system message."""
        ).format(
            err_msg=err_msg,
            exceptions_str=format_exceptions(exceptions),
            task_str=task_str,
            instruction=format_instruction(instruction),
        )

    @staticmethod
    def format_instruction_msg(
        err_msg: str,
        exceptions: list[ExceptionDetails],
        diffs_by_repo_name: list[tuple[str, str]],
        instruction: str | None,
    ):
        changes_str = ""
        for repo, diff in diffs_by_repo_name:
            changes_str += textwrap.dedent(
                """\

                <changes repo_name="{repo}">
                {diff}
                </changes>"""
            ).format(repo=repo, diff=diff)

        return textwrap.dedent(
            """\
            This is in response to the below issue:
            <issue>
            <error_message>
            {err_msg}
            </error_message>
            {exceptions_str}
            </issue>

            The following changes have been made to the codebase to fix the issue:
            {changes_str}

            You are given the following instruction in relationship to the above changes and you have to break it into steps:
            <instruction>
            {instruction}
            </instruction>

            Think step-by-step inside the <thoughts> tag then output a concise and simple list of steps to perform in the output format provided in the system message."""
        ).format(
            err_msg=err_msg,
            exceptions_str=format_exceptions(exceptions),
            changes_str=changes_str,
            instruction=instruction,
        )
