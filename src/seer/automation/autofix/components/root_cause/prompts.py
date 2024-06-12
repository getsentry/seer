import textwrap
from typing import Optional

from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPromptXml,
)
from seer.automation.autofix.prompts import format_instruction
from seer.automation.models import EventDetails, ExceptionDetails


class RootCauseAnalysisPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at finding the root cause of any issue.

            You have tools to search a codebase to find the root cause of an issue. Please use the tools as many times as you want to find the root cause of the issue. It is very important to be very detailed and clear in your output.

            You must follow the below XML format in your output:
            {root_cause_output_example_str}

            Notes:
            - The likelihood must be a float between 0 and 1. It represents the likelihood that the root cause is correct.
            - The actionability must be a float between 0 and 1. It represents if a fix to this cause is actionable within this codebase.
                - For example, if it's caused by a malformed API request, then it's not actionable in the codebase.
                - If there is a clear code change that can be made to fix the issue, then it is actionable.
                - If you do not have a clear code change that can be made to fix the issue, then it should be scored low.
            - You can include suggested fixes if you have ones that are clear and actionable.
                - In a suggested fix, suggest a fix with expert judgement, consider the implications of the fix, and the potential side effects.
                - For example, simply raising an exception or assigning a default value may not be the best fix.
                - The elegance of a fix is a float between 0 and 1. The higher the score the better the fix.
                    - A fix by a staff engineer will have a high elegance score.
                    - A fix by a junior engineer or intern will have a low elegance score.
            - Don't always assume the data being passed is correct, it could be incorrect! Sometimes the API request is malformed, or there is a bug on the client/server side that is causing the issue.

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

            Think step-by-step in a <thoughts> block before returning the potential root causes of the issue inside a <potential_root_causes> block."""
        ).format(
            error_str=event.format_event(),
            instruction_str=format_instruction(instruction),
        )
