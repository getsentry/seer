import textwrap

from seer.automation.autofix.components.coding.models import PlanStepsPromptXml
from seer.automation.autofix.prompts import format_instruction
from seer.automation.models import EventDetails


class CodingPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that solves problems with the best plans.

            You are giving tasks to a coding agent that will perform code changes based on your instructions. The tasks must be clear and detailed enough that the coding agent can perform the task without any additional information.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets. Every time you do something, explain the reason using the following sentence and say nothing else: I'll do X because Y.

            Your output must follow the format properly according to the following guidelines:

            {steps_example_str}
            """
        ).format(steps_example_str=PlanStepsPromptXml.get_example().to_prompt_str())

    @staticmethod
    def format_default_msg(event: EventDetails, task_str: str, instruction: str | None):
        return textwrap.dedent(
            """\
            Given the issue:
            {event_str}



            The root cause of the issue has been identified and a fix has been suggested. The fix is as follows:
            {task_str}

            Break down the task of fixing the issue into steps. Since you are a principal engineer, your solution should not just add logs or throw more errors, but should meaningfully fix the issue. Your list of steps should be detailed enough so that following it exactly will lead to a fully complete solution.

            Think step-by-step each time before using the tools provided to you inside a <thoughts></thoughts> block.
            Also think step-by-step inside a <thoughts></thoughts> block before giving the final answer.

            When ready with your final answer, detail the precise plan to accomplish the task wrapped with a <plan_steps></plan_steps> block.

            <guidelines>
            - Each file change must be a separate step and be explicit and clear.
              - You MUST include exact file paths for each step you provide. If you cannot, find the correct path.
            - No placeholders are allowed, the steps must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting the steps.
            - The plan must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            </guidelines>"""
        ).format(
            event_str=event.format_event(),
            task_str=task_str,
            instruction=format_instruction(instruction),
        )
