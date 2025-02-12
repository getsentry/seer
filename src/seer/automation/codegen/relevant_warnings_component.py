import logging

from cachetools import LRUCache, cached  # type: ignore[import-untyped]
from cachetools.keys import hashkey  # type: ignore[import-untyped]
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track
from tqdm.autonotebook import tqdm

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.codebase.models import SentryIssue
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    CodeAreIssuesFixableOutput,
    CodeAreIssuesFixableRequest,
    CodeRelevantWarningsOutput,
    CodeRelevantWarningsRequest,
)
from seer.automation.codegen.prompts import IsFixableIssuePrompts, ReleventWarningsPrompts
from seer.automation.component import BaseComponent
from seer.automation.models import RelevantWarningResult
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


def _is_issue_fixable_cache_key(llm_client: LlmClient, issue: SentryIssue) -> tuple[str]:
    return hashkey(issue.group_id)


@cached(cache=LRUCache(maxsize=1000), key=_is_issue_fixable_cache_key)
def is_issue_fixable(llm_client: LlmClient, issue: SentryIssue) -> bool:
    """
    Given an issue, predict whether it's fixable.
    LRU-cached by the `issue.group_id` in case the same issue is analyzed across many requests.
    """
    completion = llm_client.generate_structured(
        model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
        system_prompt=IsFixableIssuePrompts.format_system_msg(),
        prompt=IsFixableIssuePrompts.format_prompt(
            formatted_error=issue.format_error(),
        ),
        response_format=IsFixableIssuePrompts.IsIssueFixable,
        temperature=0.0,
        max_tokens=2048,
        timeout=7.0,
    )
    return completion.parsed.is_fixable


class AreIssuesFixableComponent(
    BaseComponent[CodeAreIssuesFixableRequest, CodeAreIssuesFixableOutput]
):
    """
    Given a list of issues, predict whether each is fixable.
    """

    context: CodegenContext
    _max_issues_analyzed: int = 10

    @observe(name="Predict Issue Fixability")
    @ai_track(description="Predict Issue Fixability")
    @inject
    def invoke(
        self, request: CodeAreIssuesFixableRequest, llm_client: LlmClient = injected
    ) -> CodeAreIssuesFixableOutput:
        """
        It's fine if there are duplicate issues in the request. That can happen if issues were
        passed in from a list of warning-issue associations.
        """
        # TODO: batch / send uncached issues in one prompt and ask for a list of fixable issue group ids
        issue_group_id_to_issue = {issue.group_id: issue for issue in request.candidate_issues}
        issue_group_ids = list(issue_group_id_to_issue.keys())[: self._max_issues_analyzed]
        issue_group_id_to_is_fixable = {
            issue_group_id: is_issue_fixable(llm_client, issue_group_id_to_issue[issue_group_id])
            for issue_group_id in tqdm(issue_group_ids, desc="Predicting issue fixability")
        }
        return CodeAreIssuesFixableOutput(
            are_fixable=[
                issue_group_id_to_is_fixable.get(issue.group_id)
                for issue in request.candidate_issues
            ]
        )


class RelevantWarningsComponent(
    BaseComponent[CodeRelevantWarningsRequest, CodeRelevantWarningsOutput]
):
    """
    Given a list of warning-issue associations, predict whether each is relevant.
    A warning is relevant to an issue if fixing the warning would fix the issue (according to an
    LLM).
    """

    context: CodegenContext
    _max_associations_analyzed: int = 10

    @observe(name="Predict Relevant Warnings")
    @ai_track(description="Predict Relevant Warnings")
    @inject
    def invoke(
        self, request: CodeRelevantWarningsRequest, llm_client: LlmClient = injected
    ) -> CodeRelevantWarningsOutput:
        # TODO: instead of looking at every association, probably faster and cheaper to input one
        # warning and prompt for which of its associated issues are relevant. May not work as well.
        #
        # TODO: handle LLM API errors in this loop by moving on
        relevant_warning_results: list[RelevantWarningResult] = []
        candidate_associations = request.candidate_associations[: self._max_associations_analyzed]
        # The time limit for OpenAI prompt caching is 5-10 minutes, so no point in sorting by
        # issue.group_id
        for warning, issue in tqdm(candidate_associations, desc="Predicting relevance"):
            completion = llm_client.generate_structured(
                model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
                system_prompt=ReleventWarningsPrompts.format_system_msg(),
                prompt=ReleventWarningsPrompts.format_prompt(
                    formatted_warning=warning.format_warning(),
                    formatted_error=issue.format_error(),
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
