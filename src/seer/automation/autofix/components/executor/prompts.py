import textwrap

from seer.automation.autofix.prompts import format_exceptions
from seer.automation.models import ExceptionDetails


class ExecutionPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional senior engineer that is responsible for correctly resolving a production issue. Given the available tools and below task, which corresponds to an important step in resolving the issue, convert the task into code. The original error message and stack trace that the plan is designed to address is also provided to help you understand the context of your task.
            It's absolutely vital that you completely and correctly execute your task.

            When the task is complete, reply with "<DONE>"
            If you are unable to complete the task, also reply with "<DONE>"

            <guidelines>
                - Write code by calling the available tools.
                - The code must be valid, executable code.
                - Code padding, spacing, and indentation matters, make sure that the indentation is corrected for.
                - `multi_tool_use.parallel` is invalid, do not use it.
                - You cannot call tools via XML, use the tool calling API instead.
            </guidelines>"""
        )

    @staticmethod
    def format_default_msg(
        retriever_dump: str | None,
        error_message: str | None,
        exceptions: list[ExceptionDetails],
        task: str,
    ):
        context_dump_str = (
            textwrap.dedent(
                """\
                <relevant_context>
                {retriever_dump}
                </relevant_context>"""
            ).format(retriever_dump=retriever_dump)
            if retriever_dump
            else ""
        )

        issue_str = textwrap.dedent(
            """\
                <issue>
                <error_message>
                {error_message}
                </error_message>
                {exceptions_str}
                </issue>"""
        ).format(
            error_message=error_message,
            exceptions_str=format_exceptions(exceptions),
        )

        return (
            textwrap.dedent(
                """\
            {context_dump_str}

            {issue_str}

            <task>
            {task_text}
            </task>

            You must complete the task.
            - Think out loud step-by-step before you start writing code.
            - Do not just add a comment or leave a TODO, you must write functional code.
            - Importing libraries and modules should be done in its own step.
            - Carefully review your code and ensure that it is formatted correctly.

            You must use the tools/functions provided to do so."""
            )
            .format(context_dump_str=context_dump_str, issue_str=issue_str, task_text=task)
            .strip()
        )
