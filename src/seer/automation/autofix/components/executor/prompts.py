import textwrap

from seer.automation.codebase.models import BaseDocument
from seer.automation.models import EventDetails


class ExecutionPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional senior engineer that is responsible for correctly resolving a production issue. Given the available tools and below task, which corresponds to an important step in resolving the issue, convert the task into code. The original error message and stack trace that the plan is designed to address is also provided to help you understand the context of your task. Every time you do something, explain the reason using the following sentence and say nothing else: I'll do X because Y.

            It's absolutely vital that you completely and correctly execute your task. Your final solution should be complete, functional, and ready for a PR review, so make all necessary changes.

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
        documents: list[BaseDocument],
        event: EventDetails,
        task: str,
        repo_name: str,
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

        document_contents_str = (
            "\n\n"
            + textwrap.dedent(
                """\
                <relevant_documents>
                {document_contents}
                </relevant_documents>"""
            ).format(
                document_contents="\n".join(
                    [
                        document.get_prompt_xml(repo_name=repo_name).to_prompt_str()
                        for document in documents
                    ]
                )
            )
            if documents
            else ""
        )

        return (
            textwrap.dedent(
                """\
            {context_dump_str}{document_contents_str}

            {event_str}

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
            .format(
                context_dump_str=context_dump_str,
                document_contents_str=document_contents_str,
                event_str=event.format_event(),
                task_text=task,
            )
            .strip()
        )
