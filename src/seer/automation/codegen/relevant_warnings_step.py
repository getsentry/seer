import json
import logging
from typing import Any

import requests
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from integrations.codecov.codecov_auth import get_codecov_auth_header
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
    FilterWarningsOutput,
    FilterWarningsRequest,
    PrFile,
)
from seer.automation.codegen.relevant_warnings_component import (
    AreIssuesFixableComponent,
    AssociateWarningsWithIssuesComponent,
    FetchIssuesComponent,
    FilterWarningsComponent,
    PredictRelevantWarningsComponent,
)
from seer.automation.codegen.step import CodegenStep
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

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

    @inject
    def _post_results_to_overwatch(
        self,
        relevant_warnings_output: CodePredictRelevantWarningsOutput,
        config: AppConfig = injected,
    ):
        if not self.request.should_post_to_overwatch:
            logger.info("Skipping posting relevant warnings results to Overwatch.")
            return

        request = {
            "run_id": self.context.run_id,
            "results": relevant_warnings_output.model_dump()["relevant_warning_results"],
        }
        request_data = json.dumps(request, separators=(",", ":")).encode("utf-8")
        headers = get_codecov_auth_header(
            request_data,
            signature_header="X-GEN-AI-AUTH-SIGNATURE",
            signature_secret=config.OVERWATCH_OUTGOING_SIGNATURE_SECRET,
        )
        requests.post(
            url=self.request.callback_url,
            headers=headers,
            data=request_data,
        ).raise_for_status()

    def _complete_run(
        self,
        relevant_warnings_output: CodePredictRelevantWarningsOutput,
    ):
        try:
            self._post_results_to_overwatch(relevant_warnings_output)
        except Exception:
            logger.exception("Error posting relevant warnings results to Overwatch")
            raise
        finally:
            self.context.event_manager.mark_completed_and_extend_relevant_warning_results(
                relevant_warnings_output.relevant_warning_results
            )

    @observe(name="Codegen - Relevant Warnings Step")
    @ai_track(description="Codegen - Relevant Warnings Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - Relevant Warnings Step")
        self.context.event_manager.mark_running()

        # 1. Read the commit.
        repo_client = self.context.get_repo_client(type=RepoClientType.READ)
        commit = repo_client.repo.get_commit(self.request.commit_sha)
        pr_files = [
            PrFile(
                filename=file.filename, patch=file.patch, status=file.status, changes=file.changes
            )
            for file in commit.files
            if file.patch
        ]

        # 2. Only consider warnings from files changed in the commit.
        filter_warnings_component = FilterWarningsComponent(self.context)
        filter_warnings_request = FilterWarningsRequest(
            warnings=self.request.warnings, target_filenames=[file.filename for file in pr_files]
        )
        filter_warnings_output: FilterWarningsOutput = filter_warnings_component.invoke(
            filter_warnings_request
        )
        warnings = filter_warnings_output.warnings

        if not warnings:  # exit early to avoid unnecessary issue-fetching.
            self.logger.info("No warnings to predict relevancy for.")
            self._complete_run(
                relevant_warnings_output=CodePredictRelevantWarningsOutput(
                    relevant_warning_results=[]
                ),
            )
            return

        # 3. Fetch issues related to the commit.
        fetch_issues_component = FetchIssuesComponent(self.context)
        fetch_issues_request = CodeFetchIssuesRequest(
            organization_id=self.request.organization_id, pr_files=pr_files
        )
        fetch_issues_output: CodeFetchIssuesOutput = fetch_issues_component.invoke(
            fetch_issues_request
        )

        # 4. Limit the number of warning-issue associations we analyze to the top
        #    max_num_associations.
        association_component = AssociateWarningsWithIssuesComponent(self.context)
        associations_request = AssociateWarningsWithIssuesRequest(
            warnings=warnings,
            filename_to_issues=fetch_issues_output.filename_to_issues,
            max_num_associations=self.request.max_num_associations,
        )
        associations_output: AssociateWarningsWithIssuesOutput = association_component.invoke(
            associations_request
        )
        associations = associations_output.candidate_associations

        # 5. Filter out unfixable issues b/c our definition of "relevant" is that fixing the warning
        #    will fix the issue.
        are_issues_fixable_component = AreIssuesFixableComponent(self.context)
        are_fixable_output: CodeAreIssuesFixableOutput = are_issues_fixable_component.invoke(
            CodeAreIssuesFixableRequest(
                candidate_issues=[issue for _, issue in associations],
                max_num_issues_analyzed=self.request.max_num_issues_analyzed,
            )
        )
        associations_with_fixable_issues = [
            association
            for association, is_fixable in zip(
                associations, are_fixable_output.are_fixable, strict=True
            )
            if is_fixable
        ]

        # 6. Predict which warnings are relevant to which issues.
        prediction_component = PredictRelevantWarningsComponent(self.context)
        request = CodePredictRelevantWarningsRequest(
            candidate_associations=associations_with_fixable_issues
        )
        relevant_warnings_output: CodePredictRelevantWarningsOutput = prediction_component.invoke(
            request
        )

        # 7. Save results.
        self._complete_run(relevant_warnings_output)
