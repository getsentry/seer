import textwrap
from typing import Optional

from seer.automation.autofix.components.assessment.models import ProblemDiscoveryOutput
from seer.automation.autofix.prompts import format_exceptions, format_instruction
from seer.automation.models import ExceptionDetails


class PlanningPrompts:
    @staticmethod
    def format_default_msg(
        err_msg: str,
        exceptions: list[ExceptionDetails],
        problem: ProblemDiscoveryOutput,
        instruction: Optional[str] = None,
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

            The problem has been identified as:

            {problem_description}
            {instruction_str}

            Please generate a plan that fixes the above problems. DO NOT include any unit or integration tests in the plan. The shortest, simplest plan is needed."""
        ).format(
            err_msg=err_msg,
            exceptions_str=format_exceptions(exceptions),
            problem_description=problem.description,
            instruction_str=format_instruction(instruction),
        )

    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that, given an issue in the codebase and the result of a triage, is tasked with coming up with an actionable plan to fix the issue. Given the below error message and stack trace, please output actionable coding actions in XML format inside a <plan> tag to fix the issue.
            You have the ability to look up code snippets from the codebase, and you can use the snippets to help you plan the steps needed to fix the issue.

            Assume that we are not able to execute the code or run tests, so the plan should be revolved around writing code changes.
            Include the relevant filenames and line numbers in every step of the plan; each will be sent separately to the execution system to be executed.

            <guidelines>
                - Think out loud step-by-step as you search the codebase and write the plan.
                - The plan MUST correspond to a specific series of code changes needed to resolve the issue.
                - Feel free to search around the codebase to understand the code structure of the project and context of why the issue occurred before outputting the plan.
                - Search as many times as you'd like as these searches are free.
                - To the extent possible, your plan should address the root cause of the problem.
                - Understand the context of the issue and the codebase before you start writing the plan.
                - Make sure that the code changed by the plan would work well with the rest of the codebase and would not introduce any new bugs.
                - You are responsible for research tasks and resolving ambiguity, so that the final plan is as clear and actionable as possible.
                - `multi_tool_use.parallel` is invalid, do not use it.
                - You cannot call tools via XML, use the tool calling API instead.
                - Call the tools via the tool calling API before you output the plan.
                - Each step in the plan MUST correspond to a clearly defined code change that can reasonably be executed.
            </guidelines>

            <output_guide>
            - Return your response in a <plan> tag.
            - Respond with ONLY the <plan> tag, do not return anything outside the <plan> XML tag.
            - Follow exactly the format below:
            <plan>
                <title>Fix the issue</title>
                <description>
                    The function here is not working because X Y Z
                </description>
                <steps>
                    <step title="Short title for task carding">
                        Change foo(arg1: str) to accept a second argument foo(arg1: str, arg2: int) and use arg2 in the function.
                    </step>
                    <step title="Short title for task carding">
                        Add a new function bar(arg1: str, arg2: int) that uses the new argument.
                    </step>
                </steps>
            </plan>
            </output_guide>"""
        )
