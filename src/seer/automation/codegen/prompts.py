import textwrap

from pydantic import BaseModel

from seer.automation.autofix.components.coding.models import PlanStepsPromptXml, PlanTaskPromptXml
from seer.automation.codebase.models import PrFile
from seer.automation.codegen.models import StaticAnalysisSuggestion
from seer.automation.models import FilePatch, annotate_hunks


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
    @staticmethod
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
            - Use the following JSON format for each comment:
                {{
                    "path": "{{file_name}}",
                    "body": "Your comment text here",
                    "start_line": The starting line number of the code block where you are suggesting changes. This must be strictly lower than the end line number.
                    "line": The end line number of the code block where you are suggesting changes,
                    "code_suggestion": "If you have a code suggestion, provide it here. Ensure you are properly escaping special characters"
                }}
            - Ensure each comment includes:
                - The correct file name ("{{file_name}}").
                - The specific line numbers requiring the comment.
                - Clear, professional, and actionable feedback.
            - Return all comments as a list of JSON objects, ready to be used in a GitHub pull request review.
            - Wrap the comments in a <comments> and </comments> block.
            """
        ).format(
            diff_str=diff_str,
        )

    @staticmethod
    def pr_review_formatter_msg():
        return textwrap.dedent(
            """\
            Format the comments into a list of JSON objects, ready to be used in a GitHub pull request review. Please ensure you insert a comma between each JSON object except for the last one."""
        )

    @staticmethod
    def format_pr_description_step(diff_str: str):
        return textwrap.dedent(
            """\
            You are given the below code changes as a diff:
            {diff_str}

            # Your goal:
            Provide a detailed, high-level description of what this pull request is doing. Focus on:
            - The main purpose and goal of the changes
            - Key technical changes and their impact
            - Any notable architectural decisions or patterns
            - Dependencies or interactions with other parts of the system
            - Any potential risks or considerations

            # Guidelines:
            - Be concise but comprehensive
            - Use clear, technical language
            - Focus on the "why" as much as the "what"
            - Highlight any significant changes that reviewers should pay special attention to
            - If there are multiple distinct changes, organize them logically
            """
        ).format(diff_str=diff_str)


class _RelevantWarningsPromptPrefix:
    """
    Stores common prompt prefixes to maximize prompt cache hits (for OpenAI calls).
    """

    @staticmethod
    def format_system_msg():
        # https://github.com/codecov/bug-prediction-research/blob/f79fc1e7c86f7523698993a92ee6557df8f9bbd1/src/scripts/ask_oracle.py#L79-L81
        return textwrap.dedent(
            """\
            You are a senior developer with extensive knowledge about code and static analysis warnings.
            You always consider carefully the context of the code and the error before making a decision.
            You are tasked with analyzing a single issue and providing a detailed explanation of the issue and the context in which it occurs.
            """
        )

    @staticmethod
    def format_prompt_error(formatted_error: str):
        """
        The error makes up the bulk of the prompt b/c of the stacktrace.
        Put it right after the system message so that the system message and error are cached
        together.
        """
        return textwrap.dedent(
            """\
            Here is an issue we had in our codebase:

            {formatted_error}
            """
        ).format(formatted_error=formatted_error)


class IsFixableIssuePrompts(_RelevantWarningsPromptPrefix):

    class IsIssueFixable(BaseModel):
        # No reasoning due to the number of LLM calls involved in the pipeline.
        # This should be an easier task.
        is_fixable: bool

    @staticmethod
    def format_prompt(formatted_error: str):
        # https://github.com/codecov/bug-prediction-research/blob/f79fc1e7c86f7523698993a92ee6557df8f9bbd1/src/scripts/ask_oracle.py#L86
        return textwrap.dedent(
            """\
            {error_prompt}

            Carefully analyze the issue above. Focus on the error, the stacktrace.
            Think about the context in which the error occurs. What are possible causes to it? How do you fix it?
            Answer "does this issue originate from within the application or is it caused from an external service not behaving well?"
            For example a type error or value error is likely to be caused by the application, while a 500/server error is likely to be caused by an external service.
            """
        ).format(error_prompt=_RelevantWarningsPromptPrefix.format_prompt_error(formatted_error))


class StaticAnalysisSuggestionsPrompts:

    class AnalysisAndSuggestions(BaseModel):
        analysis: str
        suggestions: list[StaticAnalysisSuggestion]

    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are an expert software engineer with deep expertise in static analysis, bug detection, and code review.
            Your role is to analyze code changes and identify potential bugs or security issues that could lead to runtime errors or exceptions.
            You have a keen eye for subtle issues that might not be immediately obvious, and you understand how different code patterns can lead to unexpected behavior.
            When reviewing code, you focus exclusively on actual bugs and security vulnerabilities, ignoring style issues or minor improvements that don't affect functionality.
            You provide precise, actionable feedback with clear severity and confidence assessments to help developers prioritize fixes.
            """
        )

    @staticmethod
    def format_prompt(diff_with_warnings: str, formatted_issues: str):
        return textwrap.dedent(
            """\
            You are given a diff block annotated with static analysis warnings which may or may not be important:

            {diff_with_warnings}

            You are also given a list of past Sentry issues that exist in the codebase close to the diff:
            {formatted_issues}

            # Your Goal:
            Carefully review the code changes in the diff, understand the context and surface any potential bugs that might be introduced by the changes. In your review focus on actual bugs. You should IGNORE code style, nit suggestions, and anything else that is not likely to cause a production issue.
            You SHOULD make suggestions based on the warnings and issues provided, as well as your own analysis of the code.
            Follow ALL the guidelines!!!

            # Guidelines:
            - Return AT MOST 5 suggestions, and AT MOST 1 suggestion per line of code.
            - Focus ONLY on _bugs_ and _security issues_.
            - Only surface issues that are caused by the code changes in the diff, or directly related to a warning.
            - Do NOT propose issues if the are outside the diff.
            - ALWAYS include the exact file path and line of the suggestion.
            - Assign a severity score and confidence score to each suggestion, from 0 to 1. The score should be granular, e.g., 0.432.
                - Severity score: 1 being "guaranteed an _uncaught_ exception will happen and not be caught by the code"; 0.5 being "an exception will happen but it's being caught" OR "an exception may happen depending on inputs"; 0 being "no exception will happen";
                - Confidence score: 1 being "I am 100%% confident that this is a bug";
            - Before giving your final answer, in the `analysis` section (500 to 1000 words), think carefully and out loud about which specific warnings caused by the code change will cause production errors.
              Apply a high level of scrutiny when thinking through whether there's a clear, explainable, and evidenced pathway between the warning and a production error.
              Our engineers hate sorting through noisy suggestions. So we're not asking if a warning or issue could hypothetically, under unknown and unevidenced circumstances, vaguely cause an error. You're more than welcome to dismiss warnings that are not going to cause errors.
              Ignore issues or warnings that aren't caused by the code change.
              Pay more attention to warnings and issues around wrong types, values, and syntax. Linter warnings about unnecessary imports, suboptimal style, etc. are rarely critical.
              Express in words what you're uncertain about, and what you're more confident about. You are more than free to point out that the warnings and issues are not clearly problematic in context of the code change, and should be ignored.
            - Each warning also indicates if it is a first occurrence or not. If it is a first occurrence then pay more attention to it. If it is not a first occurrence then you can analyze it with less emphasis.
            """
        ).format(
            diff_with_warnings=diff_with_warnings,
            formatted_issues=formatted_issues,
        )


class ReleventWarningsPrompts(_RelevantWarningsPromptPrefix):

    class DoesFixingWarningFixIssue(BaseModel):
        analysis: str
        does_fixing_warning_fix_issue: bool
        relevance_probability: float
        short_description: str | None = None
        short_justification: str | None = None

    @staticmethod
    def format_prompt(formatted_warning: str, formatted_error: str):
        # Defining relevance as "does fixing the warning fix the issue" is taken from BPR:
        # https://github.com/codecov/bug-prediction-research/blob/f79fc1e7c86f7523698993a92ee6557df8f9bbd1/src/scripts/ask_oracle.py#L137
        #
        # Simply asking for the `relevance_probability` is inspired by: https://arxiv.org/abs/2305.14975
        return textwrap.dedent(
            """\
            {error_prompt}

            Here is a warning that surfaced somewhere in our codebase:

            {formatted_warning}

            We have no idea if the warning is relevant to the issue. It could be completely irrelevant, for all we know!
            We need to know if this warning is *directly* relevant to the issue. By "directly relevant", we mean that fixing the warning would very likely prevent the issue.
            Our engineers hate sorting through noisy warnings. So we're not asking if the warning could hypothetically, under unknown and unevidenced circumstances, vaguely relate to the issue.
            We're asking if there's a clear, explainable, and evidenced relationship between the warning and the issue.
            The file locations of the warning and the issue don't have to be identical, but the warning and stacktrace must be referring to the same functions or variables; they must refer to the same core logic.
            Important: if the warning and issue don't refer to the same functions or variables, then you can immediately conclude that the warning is not directly relevant to the issue.

            Before giving your final answer, think out loud in an `analysis` section about the context in which the issue occurs, independent of the warning.
            Then think about how the warning is related and unrelated to the issue:
            - Is the warning referring to the same functions and variables as the issue?
            - How clear, explainable, and evidenced is the relationship between the warning and the issue?
            - Would fixing the warning likely resolve the issue?
            Your `analysis` should be at most 500 words. Express in words what you're uncertain about, and what you're more confident about.
            You are more than free to point out that the warning is completely irrelevant and should be ignored in the context of the issue.

            Next, give a score between 0 and 1 for how likely it is that addressing this warning would prevent the issue based on your `analysis`.
            This score, the `relevance_probability`, must be very granular, e.g., 0.32.
            Then give your final answer in `does_fixing_warning_fix_issue`. It should be true if and only if you're quite confident.

            Finally, if you believe the warning is relevant to the issue (`does_fixing_warning_fix_issue=true`), then fill in two more sections:
              - `short_description`: a short, fluff-free, information-dense description of the problem caused by not addressing the warning.
                This description must focus on the problem and not the warning itself. It should be at most 20 words.
              - `short_justification`: a short, fluff-free, information-dense summary of your analysis for why the warning is relevant to the issue. This justification should be at most 15 words.
            """
        ).format(
            error_prompt=_RelevantWarningsPromptPrefix.format_prompt_error(formatted_error),
            formatted_warning=formatted_warning,
        )


class RetryUnitTestPrompts:
    @staticmethod
    def format_continue_unit_tests_prompt(code_coverage_info: str, test_result_info: str):
        return textwrap.dedent(
            """\
            The tests you have generated so far are not sufficient to cover all the changes in the codebase. You need to continue generating unit tests to address the gaps in coverage and fix any failing tests.

            To help you with this, you have access to code coverage information at a file level attached as a JSON in addtion to test result information also in a JSON format.

            Using the information and instructions provided, update the unit tests to ensure robust code coverage as well as fix any failing tests. Use the exact same format you used previously to regenerate tests. Your changes will be appended as a new commit to the branch of the existing PR.

            Here is the code coverage information:
            {code_coverage_info}

            Here is the test result information:
            {test_result_info}
            """
        ).format(
            code_coverage_info=code_coverage_info,
            test_result_info=test_result_info,
        )


class BugPredictionPrompts:
    @classmethod
    def _focus_on_crashes(cls) -> str:
        return textwrap.dedent(
            """\
            IMPORTANT: we are **not** looking for:
            - silent failures
            - gracefully handled failures
            - security concerns
            - performance concerns.

            Focus exclusively on bugs that will cause the server to unexpectedly crash.
            Be careful about interpreting a code change that *intentionally* raises an error."""
        )

    @classmethod
    def format_system_msg(cls) -> str:
        # Lightly edited from:
        # src/seer/automation/autofix/components/root_cause/prompts.py
        return textwrap.dedent(
            """\
            You are an exceptional AI detective that is amazing at digging deep into code changes.
            {focus}

            You have tools to search repos to gather relevant information across the codebase. Please use the tools as many times as you want to gather relevant information.

            # Guidelines:
            - Your job is to simply gather all information needed to understand the code change and find critical bugs, not to propose fixes.
            - You are not able to search in external libraries. Do not attempt to search in external libraries.
            If you don't know how an external library works, either use the Google search tool, or just say you don't know how the external library works."""
        ).format(
            focus=cls._focus_on_crashes(),
        )

    @classmethod
    def _format_hunks(cls, pr_file: PrFile) -> str:
        hunks = FilePatch.to_hunks(pr_file.patch)
        hunks_with_line_numbers = "\n\n".join(annotate_hunks(hunks))
        return hunks_with_line_numbers

    @classmethod
    def _format_diff_file(
        cls, pr_file: PrFile, formatted_hunks: str | None = None, repo_full_name: str | None = None
    ) -> str:
        tag_start = f"<file><filename>{pr_file.filename}</filename>"
        tag_end = "</file>"

        repo_name_added = f" in repo {repo_full_name}" if repo_full_name is not None else ""

        if pr_file.status == "renamed":
            title = f"File {pr_file.previous_filename} was renamed to {pr_file.filename}{repo_name_added}"
            if formatted_hunks is None and pr_file.changes > 0:
                formatted_hunks = cls._format_hunks(pr_file)
            else:
                formatted_hunks = ""
        elif pr_file.status == "removed":
            title = f"File {pr_file.filename} was removed{repo_name_added}"
            formatted_hunks = ""
        else:
            title = f"Here are the changes made to file {pr_file.filename}{repo_name_added}"
            if formatted_hunks is None:
                formatted_hunks = cls._format_hunks(pr_file)

        return "\n\n".join((tag_start, title, formatted_hunks, tag_end))

    @classmethod
    def format_diff(cls, pr_files: list[PrFile], repo_full_name: str | None = None) -> str:
        body = "\n\n".join(
            cls._format_diff_file(pr_file, repo_full_name=repo_full_name) for pr_file in pr_files
        )
        return f"<diff>\n\n{body}\n\n</diff>"

    @classmethod
    def format_file_filter_prompt(
        cls,
        pr_files: list[PrFile],
        num_files_desired: int = 5,
    ) -> str:
        return textwrap.dedent(
            """\
            Here's a code change.

            {diff}

            We need you to narrow down the list of files we want to analyze for finding bugs.
            Return the top {num_files_desired} unique files we should analyze.
            We don't care to predict bugs for code that won't be run in production, e.g., test files. So please filter out test files. We want to predict bugs for files that might contain error-prone, untested code that could cause a production crash.
            For context, this is just a preprocessing step. You'll have the chance to do an extensive code search and analysis of this code change later. For now, we just want you to filter down the list of files to a more manageable number."""
        ).format(
            diff=cls.format_diff(pr_files),
            num_files_desired=num_files_desired,
        )

    @classmethod
    def format_prompt(cls, repos_str: str, diff: str) -> str:
        return textwrap.dedent(
            """
            You'll be given a code change. We're looking for bugs that might cause errors in production. We don't know if there are any. After all, most code changes are safe.
            {focus}

            It is important that you determine if the code change is making new assumptions.
            Feel free to hypothesize about a few things that might cause the code to crash.
            Also, for each potential bug you find, state what you need to know to investigate whether the code change is correct or incorrect.

            <available_repos>
            {repos_str}
            </available_repos>

            Here's the code change in question:

            {diff}
            """
        ).format(
            repos_str=repos_str,
            diff=diff,
            focus=cls._focus_on_crashes(),
        )

    @classmethod
    def format_prompt_structured_hypothesis(cls, hypothesis_unstructured: str) -> str:
        return textwrap.dedent(
            """
            You were given a code change and asked to hypothesize about potential bugs. Here's what you said:

            <what_you_said>
            {hypothesis_unstructured}
            </what_you_said>

            Please separate this information into a list of potential bugs. If some bugs seem inter-dependent, make sure to put them in the same element of the list.
            """
        ).format(
            hypothesis_unstructured=hypothesis_unstructured,
        )

    @classmethod
    def format_prompt_followup(
        cls, repos_str: str, diff: str, hypothesis_unstructured: str, hypothesis: str
    ) -> str:
        return textwrap.dedent(
            """
            You were given a code change:

            {diff}

            You were asked to hypothesize about potential bugs. Here's what you said:

            <what_you_said>
            {hypothesis_unstructured}
            </what_you_said>

            For now, focus on this potential bug:

            <hypothesis>
            {hypothesis}
            </hypothesis>

            Please search the codebase to see if there's evidence for this potential bug. Reference relevant parts of the codebase.
            You should have a pretty high bar for determining that a bug is a real threat to the system.
            Please be clear about what couldn't quite be verified, despite your best efforts.

            Make sure to think before making your conclusion.
            If, while investigating or explaining the bug, you realize that you can't actually verify it, that's totally fine. Just say so. We don't want you to over-promise.

            {focus}

            <available_repos>
            {repos_str}
            </available_repos>
            """
        ).format(
            repos_str=repos_str,
            diff=diff,
            hypothesis_unstructured=hypothesis_unstructured,
            hypothesis=hypothesis,
            focus=cls._focus_on_crashes(),
        )

    @classmethod
    def format_prompt_bug_prediction_formatter(cls, followups: list[str]) -> str:
        return textwrap.dedent(
            """
            You are a helpful assistant that extracts structured information from bug prediction analyses.
            You are given the following bug prediction analyses for a pull request.
            <followups>
            {followups}
            </followups>

            # Your goal:
            Review the bug prediction analyses and extract the information listed below.

            # Guidelines:
            - Ensure all fields are properly populated based on each analysis.
            - Use the following JSON format for each finding:
            ```json
            {{
              "title": string,            // A concise summary of the bug prediction
              "description": string,      // A detailed explanation of the potential issue. Phrase it politely as a suspicion.
              "affected_files": string[], // List of filenames affected by this bug. Include the full path
              "suggested_fix": string,    // Recommended approach to address the issue. Phrase it politely as a suggestion.
              "severity": float,         // Severity level, from 0 (no issue) to 1 (critical issue)
              "confidence": float,       // Confidence in prediction, from 0 (no confidence) to 1 (high confidence)
              "is_valid": boolean,        // Whether this prediction should be shown to users
              "code_locations": string[]  // Specific locations in the format of "/path/to/file.py:start_line~end_line". If the line numbers are unknown, just include the full path to the file.
            }}
            ```
            """
        ).format(followups="\n".join(followups))
