import textwrap
from typing import Optional

from seer.automation.autofix.models import (
    FileChange,
    PlanningOutput,
    PlanStep,
    ProblemDiscoveryOutput,
)


def format_additional_context(additional_context: str):
    return textwrap.dedent(
        f"""\
        Additional context has also been provided:
        {additional_context}"""
    )


def pad(s: str, p: str):
    return f"{p}{s}{p}"


class ProblemDiscoveryPrompt:
    @staticmethod
    def format_default_msg(additional_context: Optional[str] = None):
        additional_context_str = (
            pad(format_additional_context(additional_context), "\n") if additional_context else ""
        )
        return f"""Assess the above issue.{additional_context_str}"""

    # TODO: To be implemented with the feedback system
    # @staticmethod
    # def format_feedback_msg(message: str, previous_output: ProblemDiscoveryOutput):
    #     return textwrap.dedent(
    #         f"""\
    #         On a previous run, the following issues were found:

    #         {problem.description}

    #         However, we received feedback:

    #         {message}

    #         Please generate a new problems XML that addresses the feedback."""
    #     )

    @staticmethod
    def format_system_msg(err_msg: str, stack_str: str):
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is tasked with finding how actionable the problem is given an error message and stack trace. Think step-by-step before outputting your answer.

            An actionable problem is one where the cause is apparent within the error message and stack trace and would easily be fixed with a simple, straightforward code change. A non-actionable problem is one where the cause is not apparent and would require some investigation to fix and more than a simple, straightforward code change.
            The actionability score should also take into account whether this error should be fixed immediately or can be fixed at a later time.

            <output_guide>
                - Output a description of the problem and whether it is actionable in XML format inside a <problem> tag.
                - Provide a description of the problem inside the <description> tag.
                - Provide a <reasoning> tag with your reasoning for why the problem is actionable or not.
                - Inside the <actionability_score> tag, output a float score from 0-1.0 if the error message and stack trace is actionable and would be fixed with a simple, straightforward code change.
                - Make sure to escape any special characters in the XML.
                - Example format provided below:
                <problem>
                    <description>
                        The function here is not working because X Y Z
                    </description>
                    <reasoning>
                        This should be actionable because X Y Z
                    </reasoning>
                    <actionability_score>
                        0.8
                    </actionability_score>
                </problem>
            </output_guide>

            <error_message>
            {err_msg}
            </error_message>

            <stack_trace>
            {stack_str}
            </stack_trace>"""
        ).format(err_msg=err_msg, stack_str=stack_str)


class PlanningPrompts:
    @staticmethod
    def format_plan_item_query_system_msg():
        return textwrap.dedent(
            """\
            Given the below instruction, please output a JSON array of strings of queries that you would use to find the code that would accomplish the instruction.

            <guidelines>
            - The queries should be specific to the codebase and should be able to be used to find the code that would accomplish the instruction.
            - The queries can be both keywords and semantic queries.
            </guidelines>

            Examples are provided below:

            <instruction>
            "Rename the function `get_abc()` to `get_xyz()` in `static/app.py`."
            </instruction>
            <queries>
            ["get_abc", "static/app.py", "get_abc in static/app.py"]
            </queries>
            <instruction>
            "Find where the endpoint for `/api/v1/health` is defined in the codebase."
            </instruction>
            <queries>
            ["/api/v1/health", "health endpoint", "health api", "status check"]
            </queries>"""
        )

    @staticmethod
    def format_plan_item_query_default_msg(plan_item: PlanStep):
        return textwrap.dedent(
            """\
            <instruction>
            "{text}"
            </instruction>
            """
        ).format(text=plan_item.text)

    @staticmethod
    def format_default_msg(
        problem: ProblemDiscoveryOutput, additional_context: Optional[str] = None
    ):
        additional_context_str = (
            pad(format_additional_context(additional_context), "\n") if additional_context else ""
        )
        return textwrap.dedent(
            f"""\
            The problem has been identified as:

            {problem.description}
            {additional_context_str}

            Please generate the plan that fixes the above problems."""
        )

    # TODO: to be implemented with the feedback system
    # @staticmethod
    # def format_with_feedback_msg(
    #     message: str,
    #     problem: ProblemDiscoveryOutput,
    #     previous_output: PlanningOutput,
    #     additional_context: Optional[str] = None,
    # ):
    #     additional_context_str = (
    #         pad(format_additional_context(additional_context), "\n") if additional_context else ""
    #     )

    #     return textwrap.dedent(
    #         f"""\
    #         The problem has been identified as:

    #         {problem.description}

    #         Previously a plan was generated:

    #         {previous_output.plan}

    #         However, we received feedback:

    #         {message}
    #         {additional_context_str}

    #         Please generate a new plan that addresses the feedback."""
    #     )

    @staticmethod
    def format_system_msg(err_msg: str, stack_str: str):
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
                        Do X
                    </step>
                    <step title="Short title for task carding">
                        Do Y
                    </step>
                </steps>
            </plan>
            </output_guide>

            <error_message>
            {err_msg}
            </error_message>

            <stack_trace>
            {stack_str}
            </stack_trace>"""
        ).format(err_msg=err_msg, stack_str=stack_str)


class ExecutionPrompts:
    @staticmethod
    def format_default_msg(
        plan_item: PlanStep,
    ):
        return plan_item.text

    @staticmethod
    def format_system_msg(context_dump: str, error_message: str | None, stack_trace: str | None):
        issue_str = (
            textwrap.dedent(
                """\
                <issue>
                    <error_message>
                {error_message}
                    </error_message>
                    <stack_trace>
                {stack_trace}
                    </stack_trace>
                </issue>"""
            ).format(
                error_message=textwrap.indent(error_message, "        "),
                stack_trace=textwrap.indent(stack_trace, "        "),
            )
            if error_message and stack_trace
            else ""
        )

        return textwrap.dedent(
            """\
            {context_dump}
            --------
            You are an exceptional senior engineer that is responsible for correctly resolving a production issue. Given the available tools and below task, which corresponds to an important step in resolving the issue, convert the task into code. The original error message and stack trace that the plan is designed to address is also provided to help you understand the context of your task.
            It's absolutely vital that you completely and correctly execute your task.

            When the task is complete, reply with "<DONE>"
            If you are unable to complete the task, also reply with "<DONE>"

            <guidelines>
                - Please think out loud step-by-step before you start writing code.
                - Write code by calling the available tools.
                - The code must be valid, executable code.
                - Code padding, spacing, and indentation matters, make sure that the indentation is corrected for.
                - `multi_tool_use.parallel` is invalid, do not use it.
                - You cannot call tools via XML, use the tool calling API instead.
                - Do not just add a comment or leave a TODO, you must write functional code.
                - Carefully review your code to ensure that it is correct.
            </guidelines>{issue_str}"""
        ).format(context_dump=context_dump, issue_str=issue_str)
