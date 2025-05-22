import textwrap

from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.prompts import format_code_map, format_trace_tree
from seer.automation.models import Profile, TraceTree


class SolutionPrompts:
    @staticmethod
    def format_system_msg(repos_str: str, has_tools: bool):
        return textwrap.dedent(
            """\
            You are Seer, a powerful agentic AI debugging assistant designed by Sentry, the world's leading platform for helping developers debug their code.

            You are assisting a USER who is a developer trying to fix an ISSUE reported by Sentry. You have already found the ROOT CAUSE of the issue. The USER will provide you with important context on the ISSUE to start the session: the ROOT CAUSE and the ISSUE details from Sentry (may include a stack trace, breadcrumbs, trace, HTTP request, etc.).
            Now, you must lead the effort to fix the ISSUE.

            <tool_calling>
            {tool_calling_str}
            </tool_calling>

            <available_repos>
            {repos_str}
            </available_repos>

            <solution_guidelines>
            Your SOLUTION to the ISSUE must fit in naturally with the codebase. To do so, you must explore until you gain an understanding of the codebase and the system in which the ISSUE is occurring. You MUST find a SOLUTION that is both technically correct and naturally fits into the code and its intended outcomes.
            Simpler solutions to the ISSUE with minimal code changes are usually preferred. Do NOT propose multiple band-aid solutions and mitigation techniques. Instead, focus on the single most effective SOLUTION to the ROOT CAUSE of the ISSUE.
            Break down your SOLUTION into a concrete list of steps to take in the codebase. The USER will follow your suggestion and implement the code changes later. Your SOLUTION should fit smoothly into the codebase. For example, are there existing utils you can reuse? Does it preserve the intended behavior of the application?
            If code changes are not appropriate to fix the ISSUE, you may outline the appropriate SOLUTION steps instead.
            Touching infrastructure, dependencies, or third party libraries is almost never desirable.
            </solution_guidelines>

            Remember:
            - EVERY TIME before you use a tool, think step-by-step.
            - You also MUST think step-by-step before giving the final answer.
            - If the USER provides additional instructions or guidance throughout the conversation, you MUST pay close attention and follow it, as they know the codebase better than you do and your goal is to satisfy the USER.

            We will start by gathering all relevant context. Then when you are sure, propose the final solution plan for the ISSUE.
            """
        ).format(
            tool_calling_str=(
                "As you have no prior knowledge of the codebase, you must use the available tools to gather all necessary context in addition to the context provided by the USER. You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You also have tools to search for additional context, including trace-connected Sentry context such as spans, profiles, and connected errors. Use these as necessary to find the correct solution to the ISSUE. The best solution may lie elsewhere in the codebase than the original ISSUE or even its ROOT CAUSE."
                if has_tools
                else "You do not have to ability to gather more context at this point. You must come up with the best solution you can based on what you know so far."
            ),
            repos_str=repos_str,
        )

    @staticmethod
    def format_root_cause(root_cause: RootCauseAnalysisItem | str):
        if isinstance(root_cause, RootCauseAnalysisItem):
            return RootCausePlanTaskPromptXml.from_root_cause(root_cause).to_prompt_str()
        else:
            return root_cause

    @staticmethod
    def format_default_msg(
        *,
        event: str,
        root_cause: RootCauseAnalysisItem | str,
        original_instruction: str | None,
        code_map: Profile | None,
        trace_tree: TraceTree | None,
    ):
        return textwrap.dedent(
            """\
            Please begin by gathering all relevant context to understand how to fix the issue. {original_instruction} I have included everything I know about the Sentry issue so far below:

            <issue_details>
            <root_cause>
            {root_cause_str}
            </root_cause>

            <raw_issue_details>
            {event_str}
            </raw_issue_details>

            {code_map_str}
            {trace_tree_str}
            </issue_details>
            """
        ).format(
            event_str=event,
            root_cause_str=SolutionPrompts.format_root_cause(root_cause),
            original_instruction=original_instruction,
            code_map_str=(
                f"<map_of_relevant_code>{format_code_map(code_map)}</map_of_relevant_code>"
                if code_map
                else ""
            ),
            trace_tree_str=f"<trace>{format_trace_tree(trace_tree)}</trace>" if trace_tree else "",
        )

    @staticmethod
    def solution_formatter_msg():
        return textwrap.dedent(
            """\
            Format the discussed plan exactly into a list of steps in the plan to fix the issue. Exclude steps that are not part of the fix, such as adding tests and logs.

            For each item in the plan (where one item is one step to fix the issue):
              - Title: a complete sentence describing what needs to change to fix the issue.
              - Code Snippet and Analysis: A snippet of the code change and an explanation of the code change and the reasoning behind it. All Markdown formatted. (don't write the full code, just tiny snippets at most)
              - Is most important: whether this change is the SINGLE MOST important part of the solution.
            As a whole, this sequence of steps should tell the precise plan of how to fix the issue. You can put as few or as many steps as needed.

            Then, provide a concise summary of the solution. This summary must be less than 30 words and must be an information-dense single summary and must not contain filler words such as "The application..." or "The fix...".
              - Use a "matter of fact" tone, such as "Add correct validation of `foo` to the `process_task` function."."""
        )

    @staticmethod
    def solution_proposal_msg():
        return (
            "Now that we've gathered more context, please give me the final plan to fix the issue."
        )
