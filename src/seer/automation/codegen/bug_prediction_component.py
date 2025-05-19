import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Literal, TypeAlias

from langfuse.decorators import langfuse_context, observe

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.automation.autofix.prompts import format_repo_prompt
from seer.automation.autofix.tools.tools import BaseTools
from seer.automation.codebase.models import format_diff
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    BugPrediction,
    BugPredictorHypothesis,
    BugPredictorOutput,
    BugPredictorRequest,
    FilterFilesOutput,
    FilterFilesRequest,
    FormatterOutput,
    FormatterRequest,
)
from seer.automation.codegen.prompts import BugPredictionPrompts
from seer.automation.component import BaseComponent
from seer.dependency_injection import copy_modules_initializer, inject, injected


class FilterFilesComponent(BaseComponent[FilterFilesRequest, FilterFilesOutput]):
    """
    Return PR files which are the most error prone to keep the diff manageable for the draft agent.
    """

    context: CodegenContext

    @observe(name="Codegen - Bug Prediction - Filter Files Component")
    @inject
    def invoke(
        self, request: FilterFilesRequest, llm_client: LlmClient = injected
    ) -> FilterFilesOutput:
        pr_files_filterable = [
            pr_file for pr_file in request.pr_files if pr_file.should_show_hunks()
        ]
        pr_files_not_filterable = [
            pr_file for pr_file in request.pr_files if not pr_file.should_show_hunks()
        ]  # We can include these basically for free as context

        if len(pr_files_filterable) <= request.num_files_desired:
            # In this case, we could still filter out inconsequential files, e.g., test files.
            # Choosing to leave them in b/c they may be useful context for the draft agent.
            # Can eval this choice to see if it causes the draft agent to under-scrutinize files
            # which happen to have tests but are buggy.
            return FilterFilesOutput(pr_files=request.pr_files)

        if request.shuffle_files:
            # Avoid bias for files that are near the top of the list, usually alphabetical.
            random.Random(42).shuffle(pr_files_filterable)

        filenames = tuple(pr_file.filename for pr_file in pr_files_filterable)
        FilenameFromThisPR: TypeAlias = Literal[filenames]  # type: ignore[valid-type]

        response = llm_client.generate_structured(
            prompt=BugPredictionPrompts.format_prompt_file_filter(
                pr_files_filterable, num_files_desired=request.num_files_desired
            ),
            model=GeminiProvider.model("gemini-2.0-flash-001"),
            response_format=list[FilenameFromThisPR],
        )

        if response.parsed is None:
            self.logger.warning(
                "Failed to filter files intelligently.",
            )
            pr_files_picked = pr_files_filterable
        else:
            filenames_picked = response.parsed
            pr_files_picked = [
                pr_file for pr_file in pr_files_filterable if pr_file.filename in filenames_picked
            ]

        pr_files_picked = pr_files_picked[: request.num_files_desired]
        return FilterFilesOutput(pr_files=pr_files_picked + pr_files_not_filterable)


class BugPredictorComponent(BaseComponent[BugPredictorRequest, BugPredictorOutput]):
    """
    1. Draft bug hypotheses.
    2. Verify each concurrently, resulting in a thorough analysis of each hypothesis.
    """

    context: CodegenContext

    def _verify_hypothesis(
        self,
        tools: BaseTools,
        repos_str: str,
        diff: str,
        hypotheses_unstructured: str,
        hypothesis_num: int,
        hypothesis: BugPredictorHypothesis,
    ) -> tuple[int, str | None]:
        self.logger.info(f"Verifying hypothesis {hypothesis_num}")
        agent = LlmAgent(config=AgentConfig(interactive=False), tools=tools.get_tools())
        try:
            followup = agent.run(
                run_config=RunConfig(
                    system_prompt=BugPredictionPrompts.format_system_msg(),
                    prompt=BugPredictionPrompts.format_prompt_followup(
                        repos_str=repos_str,
                        diff=diff,
                        hypothesis_unstructured=hypotheses_unstructured,
                        hypothesis=hypothesis.content,
                    ),
                    max_iterations=32,
                    model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                    temperature=0.0,
                    run_name=f"Follow up hypothesis {hypothesis_num}",
                    reasoning_effort="medium",
                ),
            )
        except Exception:
            self.logger.exception(
                "Error following up on hypothesis", extra={"hypothesis_num": hypothesis_num}
            )
            followup = None
        else:
            self.logger.info(f"Followed up on hypothesis {hypothesis_num}")
        return hypothesis_num, followup

    @observe(name="Codegen - Bug Prediction - Bug Predictor Component")
    @inject
    def invoke(
        self, request: BugPredictorRequest, llm_client: LlmClient = injected
    ) -> BugPredictorOutput:
        langfuse_parent_trace_id = langfuse_context.get_current_trace_id()
        langfuse_parent_observation_id = langfuse_context.get_current_observation_id()

        repos_str = format_repo_prompt(readable_repos=self.context.state.get().readable_repos)
        diff = format_diff(pr_files=request.pr_files)

        self.logger.info(f"Follow along at {langfuse_context.get_current_trace_url()}")

        with BaseTools(self.context, repo_client_type=RepoClientType.READ) as tools:
            # Step 1a: draft hypotheses + further research questions.
            agent = LlmAgent(config=AgentConfig(interactive=False), tools=tools.get_tools())
            hypotheses_unstructured = agent.run(
                run_config=RunConfig(
                    system_prompt=BugPredictionPrompts.format_system_msg(),
                    prompt=BugPredictionPrompts.format_prompt_draft_hypotheses(
                        repos_str=repos_str, diff=diff
                    ),
                    max_iterations=32,
                    model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                    temperature=0.0,
                    run_name="Draft bug hypotheses",
                    reasoning_effort="low",
                ),
            )

            if hypotheses_unstructured is None:
                raise ValueError("Failed to draft bug hypotheses")

            # Step 1b: separate the unstructured hypotheses into a list.
            formatted_response = llm_client.generate_structured(
                prompt=BugPredictionPrompts.format_prompt_structured_hypothesis(
                    hypotheses_unstructured
                ),
                model=GeminiProvider.model("gemini-2.0-flash-001"),
                response_format=list[BugPredictorHypothesis],
                max_tokens=8192,
                run_name="Separate into list of hypotheses",
            )
            hypotheses = formatted_response.parsed

            if hypotheses is None:
                raise ValueError("Failed to structure hypotheses")

            self.logger.info(f"Found {len(hypotheses)} hypotheses")

            # Step 2: verify. Mine evidence for or against each hypothesis.
            verify_hypothesis = partial(
                self._verify_hypothesis, tools, repos_str, diff, hypotheses_unstructured
            )
            verify_hypothesis = observe(name="Verify Hypothesis")(verify_hypothesis)
            # Decorating at the function definition causes the run to hang when using
            # ThreadPoolExecutor.

            with ThreadPoolExecutor(
                max_workers=request.max_num_concurrent_calls, initializer=copy_modules_initializer()
            ) as executor:
                futures = [
                    executor.submit(
                        verify_hypothesis,
                        hypothesis_num,
                        hypothesis,
                        langfuse_parent_trace_id=langfuse_parent_trace_id,
                        langfuse_parent_observation_id=langfuse_parent_observation_id,
                    )
                    for hypothesis_num, hypothesis in enumerate(hypotheses, start=1)
                ]
                hypothesis_num_to_followup = dict(
                    future.result() for future in as_completed(futures)
                )

        followups = [
            hypothesis_num_to_followup[hypothesis_num]
            for hypothesis_num in sorted(hypothesis_num_to_followup.keys())
        ]
        return BugPredictorOutput(
            hypotheses_unstructured=hypotheses_unstructured,
            hypotheses=hypotheses,
            followups=followups,
        )


class FormatterComponent(BaseComponent[FormatterRequest, FormatterOutput]):
    """
    Format followups into bug predictions.
    """

    context: CodegenContext

    @observe(name="Codegen - Bug Prediction - Formatter Component")
    @inject
    def invoke(
        self, request: FormatterRequest, llm_client: LlmClient = injected
    ) -> FormatterOutput:
        followups: list[str] = [
            followup for followup in request.followups if followup is not None and followup != ""
        ]
        if not followups:
            self.logger.info("No valid followups found to format into bug predictions")
            return FormatterOutput(bug_predictions=[])

        try:
            response = llm_client.generate_structured(
                prompt=BugPredictionPrompts.format_prompt_reformat_followups(followups),
                model=GeminiProvider.model("gemini-2.0-flash-001"),
                response_format=list[BugPrediction],
                run_name="Bug Prediction Formatter",
                max_tokens=8192,
            )

            if response.parsed is None:
                self.logger.warning("Failed to extract structured information from bug prediction")
                return FormatterOutput(bug_predictions=[])

            return FormatterOutput(bug_predictions=response.parsed)

        except Exception as e:
            self.logger.error(f"Error formatting bug predictions: {e}")
            return FormatterOutput(bug_predictions=[])
