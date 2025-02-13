from itertools import product
from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.models import SentryIssue, StaticAnalysisWarning
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    CodeAreIssuesFixableOutput,
    CodeAreIssuesFixableRequest,
    CodegenRelevantWarningsRequest,
    CodeRelevantWarningsOutput,
    CodeRelevantWarningsRequest,
)
from seer.automation.codegen.relevant_warnings_component import (
    AreIssuesFixableComponent,
    RelevantWarningsComponent,
)
from seer.automation.codegen.step import CodegenStep
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def relevant_warnings_task(*args, request: dict[str, Any]):
    RelevantWarningsStep(request, DbStateRunTypes.RELEVANT_WARNINGS).invoke()


@inject
def _fetch_issues(
    organization_id: int,
    provider: str,
    external_id: str,
    filename_to_patch: dict[str, str],
    client: RpcClient = injected,
) -> dict[str, list[dict[str, Any]]] | None:
    """
    Makes a call to seer_rpc (in getsentry/sentry) to get issues related to each file.
    """
    response = client.call(
        "get_issues_related_to_file_patches",  # TODO: land this in sentry
        organization_id=organization_id,
        provider=provider,
        external_id=external_id,
        filename_to_patch=filename_to_patch,
    )
    return response


class RelevantWarningsStepRequest(PipelineStepTaskRequest, CodegenRelevantWarningsRequest):
    pass


class RelevantWarningsStep(CodegenStep[RelevantWarningsStepRequest, CodegenContext]):
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

    @staticmethod
    def _associate_warnings_with_issue(
        warnings: list[StaticAnalysisWarning], issue: list[SentryIssue]
    ):
        return product(warnings, issue)

    # TODO: is this @observe doing anything useful? This method doesn't return anything.
    @observe(name="Codegen - Relevant Warnings")
    @ai_track(description="Codegen - Relevant Warnings Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - Relevant Warnings Step")
        self.context.event_manager.mark_running()

        # 1. Fetch issues based on the PR.
        repo_client = self.context.get_repo_client(type=RepoClientType.READ)
        pr = repo_client.repo.get_pull(self.request.pr_id)
        filename_to_patch = {
            pr_file.filename: pr_file.patch
            # TODO: is this filename the same format as what open_pr_comment uses?
            # Need to ensure matchability wrt sentry stacktrace frame filenames
            for pr_file in pr.get_files()
            # TODO: limit the number of files we process
            if pr_file.status == "modified"
            # If it's added or deleted, there won't be past issues for it.
        }

        if not filename_to_patch:
            # TODO: POST something indicating this to overwatch
            self.logger.info("No modified files in PR, skipping relevant warnings step")
            return

        # TODO: consider adding warning lines to help filter out issues
        filename_to_issues = _fetch_issues(
            organization_id=self.request.organization_id,
            provider=self.context.repo.provider,
            external_id=self.context.repo.external_id,
            filename_to_patch=filename_to_patch,
        )
        if not filename_to_issues:
            # TODO: POST something indicating this to overwatch
            self.logger.info("No issues found for PR, skipping relevant warnings step")
            return

        # TODO (important): make associations based on the locations of the warnings and issues in the codebase
        # https://github.com/codecov/bug-prediction-research/blob/main/src/scripts/make_associations.py
        # associations = self._associate_warnings_with_issue(self.request.warnings, candidate_issues)
        # request = CodeRelevantWarningsRequest(candidate_associations=list(associations))

        # For now, mock this data.
        import json
        from pathlib import Path

        associations_dir = Path(__file__).parent
        with open(associations_dir / "candidate_associations_relevant.json", "r") as f:
            associations = json.load(f)
        associations = [
            (
                StaticAnalysisWarning.model_validate(association["warning"]),
                SentryIssue.model_validate(association["issue"]),
            )
            for association in associations
        ]

        # 2. Filter out unfixable issues b/c our definition of "relevant" is that fixing the warning
        #    will fix the issue.
        filterer = AreIssuesFixableComponent(self.context)
        are_fixable_output: CodeAreIssuesFixableOutput = filterer.invoke(
            CodeAreIssuesFixableRequest(candidate_issues=[issue for _, issue in associations])
        )
        associations_with_fixable_issues = [
            association
            for association, is_fixable in zip(
                associations, are_fixable_output.are_fixable, strict=True
            )
            if is_fixable
        ]

        # 3. Match warnings with issues if fixing the warning will fix the issue.
        matcher = RelevantWarningsComponent(self.context)
        request = CodeRelevantWarningsRequest(
            candidate_associations=associations_with_fixable_issues
        )
        relevant_warnings_output: CodeRelevantWarningsOutput = matcher.invoke(request)

        # TODO: POST relevant warnings to overwatch

        self.context.event_manager.mark_completed_and_extend_relevant_warning_results(
            relevant_warnings_output.relevant_warning_results
        )
