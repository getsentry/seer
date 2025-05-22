import textwrap
from typing import Literal

from seer.automation.autofix.prompts import (
    format_code_map,
    format_instruction,
    format_summary,
    format_trace_tree,
)
from seer.automation.models import Profile, TraceTree
from seer.automation.summarize.issue import IssueSummary


class RootCauseAnalysisPrompts:
    @staticmethod
    def format_system_msg(repos_str: str, mode: Literal["context", "reasoning"]):
        return textwrap.dedent(
            """\
            You are Seer, a powerful agentic AI debugging assistant designed by Sentry, the world's leading platform for helping developers debug their code.

            You are assisting a USER who is a developer trying to find the root cause for an ISSUE reported by Sentry. The USER will provide you with important context on the ISSUE to start the session: the ISSUE details from Sentry (may include a stack trace, breadcrumbs, trace, HTTP request, etc.).
            Now, you must lead the effort to find the root cause of the ISSUE.

            <tool_calling>
            {tool_calling_str}
            </tool_calling>

            <available_repos>
            {repos_str}
            </available_repos>

            <root_cause_guidelines>
            - Your job is to simply gather all information needed to understand what happened, not to propose fixes.
            - You must find the TRUE ROOT CAUSE. Do NOT give a superficial answer that's obvious from the ISSUE details, as the USER will be unhappy. Instead, keeping asking youself "why", digging deeper, and understanding how the entire system works, paying attention to oddities, flawed logic, and edge cases.
            - Once you can see evidence of the true root cause that would satisfy the USER, you may stop.
            - It is important that you gather all information needed to understand what happened, from the entry point of the code to the error.
            - Your job is NOT to find a solution to the ISSUE, only to find the root cause.
            </root_cause_guidelines>

            Remember:
            - EVERY TIME before you use a tool, think step-by-step.
            - You also MUST think step-by-step before giving the final answer.
            - If the USER provides additional instructions or guidance throughout the conversation, you MUST pay close attention and follow it, as they know the codebase better than you do and your goal is to satisfy the USER.

            {mode_str}"""
        ).format(
            repos_str=repos_str,
            tool_calling_str=(
                "You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You also have tools to search for additional context, including trace-connected Sentry context such as spans, profiles, and connected errors. Use these as necessary to find the correct root cause of the ISSUE. The true root cause may lie elsewhere in the codebase than where the original ISSUE occurred."
            ),
            mode_str=(
                "We will start by gathering all relevant context, not immediately proposing a root cause."
                if mode == "context"
                else "We will now reason over everything we have learned so far and provide the USER with the final root cause of the ISSUE."
            ),
        )

    @staticmethod
    def format_default_msg(
        event: str,
        instruction: str | None = None,
        summary: IssueSummary | None = None,
        code_map: Profile | None = None,
        trace_tree: TraceTree | None = None,
    ):
        return textwrap.dedent(
            """\
            Please begin by gathering all relevant context to understand what happened, from the entry point of the code to the error, and WHY it happened. {instruction_str}I have included everything I know about the Sentry issue so far below:

            <issue_details>
            {summary_str}
            {error_str}
            {code_map_str}
            {trace_tree_str}
            {instruction_str}
            </issue_details>"""
        ).format(
            error_str=event,
            instruction_str=format_instruction(instruction),
            summary_str=format_summary(summary),
            code_map_str=format_code_map(code_map),
            trace_tree_str=format_trace_tree(trace_tree),
        )

    @staticmethod
    def root_cause_proposal_msg():
        return textwrap.dedent(
            """\
            Based on all the information you've learned, please help me understand the root cause of the issue.
            """
        )

    @staticmethod
    def root_cause_formatter_msg():
        return textwrap.dedent(
            """\
            Write a reproduction timeline to illustrate how exactly the issue occurred and why.

            For each event:
              - Title: a complete sentence describing what happened and why it matters to the root cause of the issue. (a summary of the description)
              - Code Snippet and Analysis: any extra analysis needed and a small relevant code snippet if this is an important event. All Markdown formatted.
              - Event type: logic in the code, a human interaction, or an external system like a database, API, etc.
              - Is most important event: whether this event is the MOST important and insightfulone in the timeline to pay attention to.
            As a whole, this timeline should tell the precise story of the root cause of the issue. Starts at the entry point of the code, ends at the error.

            Then, provide a concise summary of the root cause of the issue. This summary must be less than 30 words and must be an information-dense single summary and must not contain filler words such as "The application..." or "The issue...".
              - Use a "matter of fact" tone, such as "The `process_task` function did not check if the task was already processed, due to `foo` being `bar`."."""
        )
