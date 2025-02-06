import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track
from tqdm.autonotebook import tqdm

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.codebase.models import SentryIssue, StaticAnalysisWarning
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    CodeRelevantWarningsOutput,
    CodeRelevantWarningsRequest,
    RelevantWarningResult,
)
from seer.automation.codegen.prompts import ReleventWarningsPrompts
from seer.automation.component import BaseComponent
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class RelevantWarningsComponent(
    BaseComponent[CodeRelevantWarningsRequest, CodeRelevantWarningsOutput]
):
    """
    Given a list of warning-issue associations, predict whether each is relevant.
    A warning is relevant to an issue if fixing the warning would fix the issue (according to an
    LLM).
    """

    context: CodegenContext
    _max_warning_issue_pairs_analyzed: int = 10

    @staticmethod
    def _format_warnings(warnings: list[StaticAnalysisWarning]) -> dict[int, str]:
        warning_id_to_formatted_warning: dict[int, str] = {}
        for warning in warnings:
            if warning.id not in warning_id_to_formatted_warning:
                warning_id_to_formatted_warning[warning.id] = warning.format_warning()
        return warning_id_to_formatted_warning

    @staticmethod
    def _format_issues(issues: list[SentryIssue]) -> dict[int, str]:
        issue_group_id_to_formatted_error: dict[int, str] = {}
        for issue in issues:
            if issue.group_id not in issue_group_id_to_formatted_error:
                issue_group_id_to_formatted_error[issue.group_id] = issue.format_error()
        return issue_group_id_to_formatted_error

    @observe(name="Predict Relevant Warnings")
    @ai_track(description="Predict Relevant Warnings")
    @inject
    def invoke(
        self, request: CodeRelevantWarningsRequest, llm_client: LlmClient = injected
    ) -> CodeRelevantWarningsOutput:
        candidate_warnings = [warning for warning, _ in request.candidate_associations]
        candidate_issues = [issue for _, issue in request.candidate_associations]

        # Format all events and warnings once since each could be part of multiple associations
        warning_id_to_formatted_warning = self._format_warnings(candidate_warnings)
        issue_group_id_to_formatted_error = self._format_issues(candidate_issues)

        # TODO (important): filter out unfixable issues

        # TODO: instead of looking at every association, probably faster and cheaper to input one
        # warning and prompt for which of its associated issues are relevant. May not work as well.
        #
        # TODO: handle LLM API errors in this loop by moving on
        relevant_warning_results: list[RelevantWarningResult] = []
        for warning, issue in tqdm(
            request.candidate_associations, desc="Predicting warning-issue relevance"
        ):
            completion = llm_client.generate_structured(
                model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
                system_prompt=ReleventWarningsPrompts.format_system_msg(),
                prompt=ReleventWarningsPrompts.format_prompt(
                    formatted_warning=warning_id_to_formatted_warning[warning.id],
                    formatted_error=issue_group_id_to_formatted_error[issue.group_id],
                ),
                response_format=ReleventWarningsPrompts.DoesFixingWarningFixIssue,
                temperature=0.0,
                max_tokens=2048,
                timeout=7.0,
            )
            relevant_warning_results.append(
                RelevantWarningResult(
                    warning_id=warning.id,
                    issue_group_id=issue.group_id,
                    does_fixing_warning_fix_issue=completion.parsed.does_fixing_warning_fix_issue,
                    relevance_probability=completion.parsed.relevance_probability,
                    reasoning=completion.parsed.reasoning,
                )
            )

        return CodeRelevantWarningsOutput(relevant_warning_results=relevant_warning_results)
