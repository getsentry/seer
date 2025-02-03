import textwrap
from typing import Optional

from seer.automation.autofix.prompts import (
    format_code_map,
    format_instruction,
    format_repo_names,
    format_summary,
)
from seer.automation.models import Profile
from seer.automation.summarize.issue import IssueSummary


class RootCauseAnalysisPrompts:
    @staticmethod
    def format_system_msg(has_tools: bool = True):
        return textwrap.dedent(
            f"""\
            You are an exceptional principal engineer that is amazing at finding the root cause of any issue.

            {
                "You have tools to search a codebase to find the root cause of an issue. Please use the tools as many times as you want to find the root cause of the issue."
                if has_tools
                else ""
            }

            # Guidelines:
            - Don't always assume data being passed is correct, it could be incorrect! Sometimes the API request is malformed, or there is a bug on the client/server side that is causing the issue.
            {"- You are not able to search in or make changes to external libraries. If the error is caused by an external library or the stacktrace only contains frames from external libraries, do not attempt to search in external libraries."
                if has_tools
                else ""
            }
            {"- At any point, please feel free to ask your teammates (who are much more familiar with the codebase) any specific questions that would help you in your analysis."
                if has_tools
                else ""
            }
            {"- At any point, please feel free to Google for information that would help you in your analysis, using the tool provided."
                if has_tools
                else ""
            }
            - If you are not able to find any potential root causes, return only <NO_ROOT_CAUSES> followed by a specific 10-20 word reason for why.
            {"- If multiple searches turn up no viable results, you should conclude the session."
                if has_tools
                else ""
            }
            - At EVERY step of your investigation, you MUST think out loud! Share what you're learning and thinking along the way, EVERY TIME YOU SPEAK.

            It is important that you trace the true root cause of this issue, from the entry point of the code to the error."""
        )

    @staticmethod
    def format_default_msg(
        event: str,
        repo_names: list[str],
        instruction: Optional[str] = None,
        summary: Optional[IssueSummary] = None,
        code_map: Optional[Profile] = None,
    ):
        return textwrap.dedent(
            """\
            {repo_names_str}
            Given the issue: {summary_str}
            {error_str}

            {code_map_str}

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
            code_map_str=format_code_map(code_map),
        )

    @staticmethod
    def root_cause_formatter_msg():
        return textwrap.dedent(
            """\
            Write a reproduction timeline to illustrate how exactly the issue occurred and why.

            For each event:
              - Title: a complete sentence describing what happened and why it matters to the root cause of the issue. (a summary of the description)
              - Code Snippet and Analysis: any extra analysis needed and a small relevant code snippet if this is an important event. All Markdown formatted.
              - Event type: logic in the code, a human interaction, an API call, data from the database, or an environment/infra factor.
              - Is most important event: whether this event is the MOST important and insightfulone in the timeline to pay attention to.
            As a whole, this timeline should tell the precise story of the root cause of the issue. Starts at the entry point of the code, ends at the error."""
        )
