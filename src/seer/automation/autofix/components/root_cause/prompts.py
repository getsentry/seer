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
            - At any point, please feel free to ask your teammates (who are much more familiar with the codebase) any specific questions that would help you in your analysis.
            - If you are not able to find any potential root causes, return only <NO_ROOT_CAUSES>.
            - If multiple searches turn up no viable results, you should conclude the session.
            - At EVERY step of your investigation, you MUST think out loud! Share what you're learning and thinking along the way, EVERY TIME YOU SPEAK.

            It is important that we find the potential root causes of the issue."""
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
            When ready with your final answer, detail the potential root cause of the issue.

            # Guidelines:
            - The root cause should be inside its own <root_cause> block.
            - Include a title and description in the root cause. Your description may be as long as you need to help your team understand the issue, explaining the issue, the root cause, why this is happening, and how you came to your conclusion.
            - In the root cause, provide snippets of the original code, each with their own titles and descriptions, to highlight where and why the issue is occurring so that your colleagues fully understand the root cause. Provide as many snippets as you want. Within your snippets, you may highlight specific lines with a comment beginning with ***.
            - You MUST include the EXACT file name and repository name in the code snippets you provide. If you cannot, do not provide a code snippet.
            - At EVERY step of your investigation, you MUST think out loud! Share what you're learning and thinking along the way, EVERY TIME YOU SPEAK."""
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
            Given all the above potential root causes you just gave, please provide 1-2 sentence instructions on how to reproduce the issue, then turn it into a concise unit test that tests for the issue for each root cause.
            - This test should intentionally fail if the issue is not fixed. It will pass if the issue is fixed.
            - Look through the codebase to find the most relevant tests to the root cause. Make sure you follow any existing testing framework and patterns.
            - For the reproduction instructions, assume the user is an experienced developer well-versed in the codebase, simply give a concise explanation of how to reproduce the issue.
            - Do not mention the unit test in the reproduction instructions, they are separate.
            - You must use the local variables provided to you in the stacktrace to give your reproduction steps.
            - Try to be open ended to allow for the most flexibility in reproducing the issue. Avoid being too confident.
            - This step is optional, if you're not sure about the reproduction steps for a root cause, just skip it."""
        )
