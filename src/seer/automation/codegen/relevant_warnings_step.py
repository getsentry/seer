import logging
from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.models import (
    AssociateWarningsWithIssuesOutput,
    AssociateWarningsWithIssuesRequest,
    CodeAreIssuesFixableOutput,
    CodeAreIssuesFixableRequest,
    CodeFetchIssuesOutput,
    CodeFetchIssuesRequest,
    CodegenRelevantWarningsRequest,
    CodePredictRelevantWarningsOutput,
    CodePredictRelevantWarningsRequest,
    PrFile,
)
from seer.automation.codegen.relevant_warnings_component import (
    AreIssuesFixableComponent,
    AssociateWarningsWithIssuesComponent,
    FetchIssuesComponent,
    PredictRelevantWarningsComponent,
)
from seer.automation.codegen.step import CodegenStep
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes

logger = logging.getLogger(__name__)


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def relevant_warnings_task(*args, request: dict[str, Any]):
    RelevantWarningsStep(request, DbStateRunTypes.RELEVANT_WARNINGS).invoke()


class RelevantWarningsStepRequest(PipelineStepTaskRequest, CodegenRelevantWarningsRequest):
    pass


class RelevantWarningsStep(CodegenStep):
    """
    Predicts which static analysis warnings in a pull request are relevant to a past Sentry issue.
    """

    name = "RelevantWarningsStep"
    request: RelevantWarningsStepRequest
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> RelevantWarningsStepRequest:
        return RelevantWarningsStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return relevant_warnings_task

    # TODO: is this @observe doing anything useful? This method doesn't return anything.
    @observe(name="Codegen - Relevant Warnings")
    @ai_track(description="Codegen - Relevant Warnings Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - Relevant Warnings Step")
        self.context.event_manager.mark_running()

        # 1. Fetch issues related to the PR.
        repo_client = self.context.get_repo_client(type=RepoClientType.READ)
        pr = repo_client.repo.get_pull(self.request.pr_id)
        fetch_issues_component = FetchIssuesComponent(self.context)
        pr_files = [
            PrFile(
                filename=file.filename, patch=file.patch, status=file.status, changes=file.changes
            )
            for file in pr.get_files()
        ]
        fetch_issues_request = CodeFetchIssuesRequest(pr_files=pr_files)
        fetch_issues_output: CodeFetchIssuesOutput = fetch_issues_component.invoke(
            fetch_issues_request
        )
        if not fetch_issues_output.filename_to_issues:
            logger.info("No issues found in PR.")
            return

        # 2. Limit the number of warning-issue associations we analyze to the top k.
        association_component = AssociateWarningsWithIssuesComponent(self.context)
        associations_request = AssociateWarningsWithIssuesRequest(
            warnings=self.request.warnings,
            filename_to_issues=fetch_issues_output.filename_to_issues,
            max_num_associations=10,
        )
        associations_output: AssociateWarningsWithIssuesOutput = association_component.invoke(
            associations_request
        )
        associations = associations_output.candidate_associations

        # 3. Filter out unfixable issues b/c our definition of "relevant" is that fixing the warning
        #    will fix the issue.
        are_issues_fixable_component = AreIssuesFixableComponent(self.context)
        are_fixable_output: CodeAreIssuesFixableOutput = are_issues_fixable_component.invoke(
            CodeAreIssuesFixableRequest(
                candidate_issues=[issue for _, issue in associations], max_issues_analyzed=10
            )
        )
        associations_with_fixable_issues = [
            association
            for association, is_fixable in zip(
                associations, are_fixable_output.are_fixable, strict=True
            )
            if is_fixable
        ]

        # 4. Match warnings with issues if fixing the warning will fix the issue.
        prediction_component = PredictRelevantWarningsComponent(self.context)
        request = CodePredictRelevantWarningsRequest(
            candidate_associations=associations_with_fixable_issues
        )
        relevant_warnings_output: CodePredictRelevantWarningsOutput = prediction_component.invoke(
            request
        )

        # TODO: POST relevant warnings to overwatch

        self.context.event_manager.mark_completed_and_extend_relevant_warning_results(
            relevant_warnings_output.relevant_warning_results
        )
