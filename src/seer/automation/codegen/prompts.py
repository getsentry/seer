import textwrap

from seer.automation.autofix.components.coding.models import PlanStepsPromptXml, PlanTaskPromptXml


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
    def format_plan_step_msg(
        diff_str: str,
        has_coverage_info: str | None = None,
        has_test_result_info: str | None = None,
    ):
        base_msg = textwrap.dedent(
            """\
            You are given the below code changes as a diff:
            {diff_str}

            # Your goal:
            Provide the most actionable and effective steps to add unit tests to ensure test coverage for all the changes in the diff.

            Since you are an exceptional principal engineer, your unit tests should not just add trivial tests, but should add meaningful ones that test all changed functionality. Your list of steps should be detailed enough so that following it exactly will lead to complete test coverage of the code changed in the given diff.

            When ready with your final answer, detail the precise plan to add unit tests.

            # Guidelines:
            - No placeholders are allowed, the unit test must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting your suggested fix.
            - The unit tests must be comprehensive. Do not provide temporary examples, placeholders, or incomplete ones.
            - In your suggested unit tests, whenever you are providing code, provide explicit diffs to show the exact changes that need to be made.
            - All your changes should be in test files.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(diff_str=diff_str)

        if has_coverage_info:
            coverage_info_msg = textwrap.dedent(
                """\
                You are also given the following code coverage information for the current diff as a JSON object:
                {coverage_info_str}

                Remember, the goal is not just to improve coverage numbers but to verify the behavior of the code meaningfully, focusing on the recent changes.
                Integrate this information with your diff analysis to provide a comprehensive and targeted testing strategy.
                """
            ).format(coverage_info_str=has_coverage_info)
            base_msg += "\n\n" + coverage_info_msg

        if has_test_result_info:
            test_result_info_msg = textwrap.dedent(
                """\
                You are provided with the following test result data for existing tests related to the diff:
                {test_result_data}

                Use this information to enhance your test creation strategy, ensuring new tests reinforce areas of failure and improve overall test suite effectiveness in the context of the introduced changes.
                """
            ).format(test_result_data=has_test_result_info)
            base_msg += "\n\n" + test_result_info_msg

        return base_msg

    @staticmethod
    def format_find_unit_test_pattern_step_msg(diff_str: str):
        return textwrap.dedent(
            """\
            You are given the below code changes as a diff:
            {diff_str}

            # Your goal:
            Look at existing unit tests in the code and succinctly describe, in clear terms, the main highlights of how they are to designed.

            Since you are an exceptional principal engineer, your description should not be trivial. Your description should be detailed enough so that following it exactly will lead to writing good and executable unit tests that follow the same design pattern.

            # Guidelines:
            - You do not have to explain each test and what it is testing. Just identify the basic libraries used as well as how the tests are structured.
            - If the codebase has no relevant tests then return the exact phrase "No relevant tests in the codebase"
            - Make sure you use the tools provided to look through the codebase and at the files that contain existing unit tests, even if they are not fully related to the changes in the given diff.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(
            diff_str=diff_str,
        )

    @staticmethod
    def format_unit_test_msg(diff_str, test_design_hint):
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

            You are also given the following test design guidelines:
            {test_design_hint}

            # Your goal:
            Write unit tests that cover the changes in the diff. You should first explain in clear and definite terms what you are adding. Then add the unit test such that lines modified, added or deleted are covered. Create multiple test files if required and cover code changed in all the files.

            When ready with your final answer, detail the explanation of the test wrapped with a <explanation></explanation> block. Your output must follow the format properly according to the following guidelines:

            {steps_example_str}

            # Guidelines:
            _ Closely follow the guidelines provided to design the tests
            - Each file change must be a separate step and be explicit and clear.
            - Before adding new files check if a file exists with same name and if it can be edited instead
            - You MUST include exact file paths for each step you provide. If you cannot, find the correct path.
            - No placeholders are allowed, the steps must be clear and detailed.
            - Make sure you use the tools provided to look through the codebase and at the files you are changing before outputting the steps.
            - The plan must be comprehensive. Do not provide temporary examples, placeholders or incomplete steps.
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        ).format(
            diff_str=diff_str,
            test_design_hint=test_design_hint,
            steps_example_str=prompt_obj.to_prompt_str(),
        )


class CodingCodeReviewPrompts:
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that provides robust and meaningful pull request comments given a change request against codebases.

            You have access to tools that allow you to search a codebase to find the relevant code snippets and view relevant files. You can use these tools as many times as you want to find the relevant code snippets.

            # Guidelines:
            - EVERY TIME before you use a tool, think step-by-step each time before using the tools provided to you.
            - You also MUST think step-by-step before giving the final answer."""
        )

    @staticmethod
    def format_pr_review_plan_step(diff_str: str):
        return textwrap.dedent(
            """\
            You are given the below code changes as a diff:
            {diff_str}

            # Your goal:
            Review the code changes in the diff and provide constructive feedback and suggestions for improvement.

            As an exceptional principal engineer, your feedback should be insightful and actionable, focusing on code quality, performance, security, and adherence to best practices.

            # Guidelines:
            - Carefully analyze the code changes and understand the context.
            - Provide specific comments on lines or sections where improvements can be made.
            - Highlight any potential bugs, performance issues, or security vulnerabilities.
            - Suggest best practices and coding standards that should be followed.
            - Be respectful and professional in your feedback.
            - Do not include any placeholders; your feedback should be clear and detailed.
            - Before giving your final answer, think step-by-step to ensure your review is thorough.
            """
        ).format(
            diff_str=diff_str,
        )

    # call this right before final_response
    def format_pr_review_messages_json():
        return ""
