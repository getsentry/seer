import textwrap

from seer.automation.autofix.components.coding.models import PlanStepsPromptXml, PlanTaskPromptXml
from seer.automation.autofix.prompts import format_instruction


class CodingPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at finding and fixing issues in codebases.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets.

            # Guidelines:
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        )

    @staticmethod
    def format_fix_discovery_msg(event: str, task_str: str, instruction: str | None):
        return textwrap.dedent(
            """\
            Given the issue:
            {event_str}

            The root cause of the issue has been identified and context about the issue has been provided:
            {task_str}

            # Your goal:
            Provide the most actionable and effective steps to fix the issue.

            Since you are an exceptional principal engineer, your solution should not just add logs or throw more errors, but should meaningfully fix the issue. Your list of steps to fix the problem should be detailed enough so that following it exactly will lead to a fully complete solution.

            When ready with your final answer, detail the precise plan to fix the issue.

            # Guidelines:
            - No placeholders are allowed, the fix must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting your suggested fix.
            - The fix must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - If the issue occurs in multiple places or files, make sure to provide a fix for each occurrence, no matter how many there are.
            - In your suggested fixes, whenever you are providing code, provide explicit diffs to show the exact changes that need to be made.
            - You do not need to make changes in test files, someone else will do that.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(
            event_str=event,
            task_str=task_str,
            instruction=format_instruction(instruction),
        )

    @staticmethod
    def format_fix_msg():
        return textwrap.dedent(
            """\
            Break down the task of fixing the issue into steps. Your list of steps should be detailed enough so that following it exactly will lead to a fully complete solution.

            When ready with your final answer, detail the precise plan to accomplish the task wrapped with a <plan_steps></plan_steps> block. Your output must follow the format properly according to the following guidelines:

            {steps_example_str}

            # Guidelines:
            - Each file change must be a separate step and be explicit and clear.
              - You MUST include exact file paths for each step you provide. If you cannot, find the correct path.
            - No placeholders are allowed, the steps must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting the steps.
            - The plan must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(steps_example_str=PlanStepsPromptXml.get_example().to_prompt_str())


class CodingUnitTestPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at writing unit tests given a change request against codebases.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets.

            # Guidelines:
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        )

    @staticmethod
    def format_unit_test_msg(diff_str):
        example = PlanTaskPromptXml(
            file_path="path/to/file.py",
            repo_name="owner/repo",
            type="Either 'file_change', 'file_create', or 'file_delete'",
            description="Describe what you are doing here in detail like you are explaining it to a software engineer.",
            diff=textwrap.dedent(
                """\
                # Here provide the EXACT unified diff of the code change required to accomplish this step.
                # You must prefix lines that are removed with a '-' and lines that are added with a '+'. Context lines around the change are required and must be prefixed with a space.
                # Make sure the diff is complete and the code is EXACTLY matching the files you see.
                # For example:
                --- a/path/to/file.py
                +++ b/path/to/file.py
                @@ -1,3 +1,3 @@
                    return 'fab'
                    y = 2
                    x = 1
                -def foo():
                +def foo():
                    return 'foo'
                    def bar():
                    return 'bar'
                """
            ),
            commit_message="Provide a commit message that describes the unit test you are adding or changing",
        )

        prompt_obj = PlanStepsPromptXml(
            tasks=[
                example,
                example,
            ]
        )

        return textwrap.dedent(
            """\
            You are given a code diff:
            {diff_str}

            # Your goal:
            Write unit tests for the lines that have been identified as lacking test coverage. You should first explain in clear and definite terms what you are adding. Then add the unit test so that the lines that initially missed test coverage have it.

            When ready with your final answer, detail the explanation of the test wrapped with a <explanation></explanation> block. Your output must follow the format properly according to the following guidelines:

            {steps_example_str}

            # Guidelines:
            - Each file change must be a separate step and be explicit and clear.
              - You MUST include exact file paths for each step you provide. If you cannot, find the correct path.
            - No placeholders are allowed, the steps must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting the steps.
            - The plan must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(diff_str=diff_str, steps_example_str=prompt_obj.to_prompt_str())
