import textwrap
from typing import Optional

from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPromptXml,
)
from seer.automation.autofix.prompts import format_instruction
from seer.automation.models import EventDetails


class RootCauseAnalysisPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at finding the root cause of any issue.

            You have tools to search a codebase to find the root cause of an issue. Please use the tools as many times as you want to find the root cause of the issue.
            Every time you use a tool, explain the reason using the following sentence and say nothing else: I'll do X because Y.

            Guidelines:
            - Don't always assume data being passed is correct, it could be incorrect! Sometimes the API request is malformed, or there is a bug on the client/server side that is causing the issue.
            - You are not able to search in or make changes to external libraries. If the error is caused by an external library or the stacktrace only contains frames from external libraries, do not attempt to search in external libraries.
            - If you are not able to find any potential root causes, return only <NO_ROOT_CAUSES>.
            - If multiple searches turn up no viable results, you should conclude the session.

            It is important that we find all the potential root causes of the issue, so provide as many possibilities as you can for the root cause, ordered from most likely to least likely."""
        ).format(
            root_cause_output_example_str=MultipleRootCauseAnalysisOutputPromptXml.get_example().to_prompt_str(),
        )

    @staticmethod
    def format_default_msg(
        event: EventDetails,
        instruction: Optional[str] = None,
    ):
        return textwrap.dedent(
            """\
            Given the issue:
            {error_str}

            {instruction_str}

            Think step-by-step each time before using the tools provided to you.
            Also think step-by-step before giving the final answer.

            When ready with your final answer, detail all the potential root causes of the issue inside wrapped with a <potential_root_causes></potential_root_causes> block.
            - Each root cause should be inside its own <root_cause> block.
            - Include a title and description in each root cause.
            - Include float values from 0.0-1.0 of the likelihood and actionability of each root cause.
            - In each root cause, provide snippets of the original code, each with their own titles and descriptions, to highlight where and why the issue is occurring so that your colleagues fully understand the root cause. Provide as many snippets as you want. Within your snippets, you may highlight specific lines with a comment beginning with ***.
            - You MUST include the EXACT file name in the code snippets you provide. If you cannot, do not provide a code snippet.
            - To maintain valid XML format, escape any special characters in-between tags, such as &lt; for < and &gt; for >."""
        ).format(
            error_str=event.format_event(),
            instruction_str=format_instruction(instruction),
        )

    @staticmethod
    def root_cause_formatter_msg(raw_root_cause_output: str):
        return textwrap.dedent(
            """\
            Given the following root cause analysis:

            {raw_root_cause_output}

            Please format the XML properly to match the following example:

            {root_cause_output_example_str}

            Note: If the provided root cause analysis is not formatted properly, such as code snippets missing descriptions, you can derive them from the provided root cause analysis.

            Return only the formatted root cause analysis:"""
        ).format(
            raw_root_cause_output=raw_root_cause_output,
            root_cause_output_example_str=MultipleRootCauseAnalysisOutputPromptXml.get_example().to_prompt_str(),
        )
