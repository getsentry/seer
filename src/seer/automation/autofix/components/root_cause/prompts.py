import textwrap
from typing import Optional

from seer.automation.autofix.prompts import format_instruction, format_repo_names, format_summary
from seer.automation.summarize.issue import IssueSummary


class RootCauseAnalysisPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at finding the root cause of any issue.

            You have tools to search a codebase to find the root cause of an issue. Please use the tools as many times as you want to find the root cause of the issue.

            # Guidelines:
            - Don't always assume data being passed is correct, it could be incorrect! Sometimes the API request is malformed, or there is a bug on the client/server side that is causing the issue.
            - You are not able to search in or make changes to external libraries. If the error is caused by an external library or the stacktrace only contains frames from external libraries, do not attempt to search in external libraries.
            - If you are not able to find any potential root causes, return only <NO_ROOT_CAUSES>.
            - If multiple searches turn up no viable results, you should conclude the session.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer.

            It is important that we find all the potential root causes of the issue, so provide as many possibilities as you can for the root cause, ordered from most likely to least likely."""
        )

    @staticmethod
    def format_default_msg(
        event: str,
        repo_names: list[str],
        instruction: Optional[str] = None,
        summary: Optional[IssueSummary] = None,
    ):
        return textwrap.dedent(
            """\
            {repo_names_str}
            Given the issue: {summary_str}
            {error_str}

            {instruction_str}
            When ready with your final answer, detail all the potential root causes of the issue.

            # Guidelines:
            - Each root cause should be inside its own <root_cause> block.
            - Include a title and description in each root cause. Your description may be as long as you need to help your team understand the issue, explaining the issue, the root cause, why this is happening, and how you came to your conclusion.
            - Include float values from 0.0-1.0 of the likelihood and actionability of each root cause.
            - In each root cause, provide snippets of the original code, each with their own titles and descriptions, to highlight where and why the issue is occurring so that your colleagues fully understand the root cause. Provide as many snippets as you want. Within your snippets, you may highlight specific lines with a comment beginning with ***.
            - You MUST include the EXACT file name and repository name in the code snippets you provide. If you cannot, do not provide a code snippet.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(
            error_str=event,
            repo_names_str=format_repo_names(repo_names),
            instruction_str=format_instruction(instruction),
            summary_str=format_summary(summary),
        )

    @staticmethod
    def root_cause_formatter_msg():
        return textwrap.dedent(
            """\
            Please format the output properly.

            Note: If the provided root cause analysis is not formatted properly, such as code snippets missing descriptions, you can derive them from the provided root cause analysis.

            Return only the formatted root cause analysis:"""
        )

    @staticmethod
    def reproduction_prompt_msg():
        return textwrap.dedent(
            """\
            Given all the above potential root causes you just gave, please provide a 1-2 sentence concise instruction on how to reproduce the issue for each root cause.
            - Assume the user is an experienced developer well-versed in the codebase, simply give the reproduction steps.
            - You must use the local variables provided to you in the stacktrace to give your reproduction steps.
            - Try to be open ended to allow for the most flexibility in reproducing the issue. Avoid being too confident.
            - This step is optional, if you're not sure about the reproduction steps for a root cause, just skip it."""
        )
