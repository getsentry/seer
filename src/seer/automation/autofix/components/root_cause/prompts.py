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

            You have tools to search a codebase to find the root cause of an issue. Please use the tools as many times as you want to find the root cause of the issue. Every time you use a tool, you need to justify why extremely quickly to avoid being fired; to do so, just fill in this one sentence and say nothing else: I'll do X because Y.

            Guidelines:
            - Don't always assume data being passed is correct, it could be incorrect! Sometimes the API request is malformed, or there is a bug on the client/server side that is causing the issue.
            - You are not able to search in or make changes to external libraries. If the error is caused by an external library or the stacktrace only contains frames from external libraries, do not attempt to search or suggest changes in external libraries.
            - You must only suggest actionable fixes that can be made in the immediate workable codebase. Do not suggest fixes with code snippets in external libraries.
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
            - Each root cause should be inside its own <root_cause> block with its suggested fix in there too.
            - Include float values from 0.0-1.0 of the likelihood and actionability of each root cause.
            - If there is a clear and obvious fix to a given root cause, suggest a fix and provide a code snippet if possible. Suggest as many fixes as you can.
            - You MUST include the EXACT file name in the code snippets you provide. If you cannot, do not provide a code snippet."""
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

            Please format properly according to the following guidelines:

            {root_cause_output_example_str}

            Note: If the provided root cause analysis is not formatted properly, such as suggested fixes missing descriptions, you can derive them from the provided root cause analysis.

            Return only the formatted root cause analysis:"""
        ).format(
            raw_root_cause_output=raw_root_cause_output,
            root_cause_output_example_str=MultipleRootCauseAnalysisOutputPromptXml.get_example().to_prompt_str(),
        )
