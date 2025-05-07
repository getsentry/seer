import itertools
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
from seer.automation.codebase.models import PrFile
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.models import (
    AssociateWarningsWithIssuesOutput,
    AssociateWarningsWithIssuesRequest,
    CodeAreIssuesFixableOutput,
    CodeAreIssuesFixableRequest,
    CodeFetchIssuesOutput,
    CodeFetchIssuesRequest,
    CodegenRelevantWarningsRequest,
    CodePredictStaticAnalysisSuggestionsOutput,
    CodePredictStaticAnalysisSuggestionsRequest,
    FilterWarningsOutput,
    FilterWarningsRequest,
)
from seer.automation.codegen.relevant_warnings_component import (
    AreIssuesFixableComponent,
    AssociateWarningsWithIssuesComponent,
    FetchIssuesComponent,
    FilterWarningsComponent,
    StaticAnalysisSuggestionsComponent,
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
    Predicts which static analysis warnings in a pull request are relevant to a past Sentry issue
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
        llm_suggestions: CodePredictStaticAnalysisSuggestionsOutput | None,
        diagnostics: list | None,
        config: AppConfig = injected,
    ):

        if not self.request.should_post_to_overwatch:
            self.logger.info("Skipping posting relevant warnings results to Overwatch.")
            return

        # This should be a temporary solution until we can update
        # Overwatch to accept the new format.
        suggestions_to_overwatch_expected_format = (
            [
                suggestion.to_overwatch_format().model_dump()
                for suggestion in llm_suggestions.suggestions
            ]
            if llm_suggestions
            else []
        )

        request = {
            "run_id": self.context.run_id,
            "results": suggestions_to_overwatch_expected_format,
            "diagnostics": diagnostics or [],
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
        static_analysis_suggestions_output: CodePredictStaticAnalysisSuggestionsOutput | None,
        diagnostics: list | None,
    ):
        try:
            self._post_results_to_overwatch(static_analysis_suggestions_output, diagnostics)
        except Exception:
            self.logger.exception("Error posting relevant warnings results to Overwatch")
            raise
        finally:
            self.context.event_manager.mark_completed_and_extend_static_analysis_suggestions(
                static_analysis_suggestions_output.suggestions
                if static_analysis_suggestions_output
                else []
            )

    @observe(name="Codegen - Relevant Warnings Step")
    @ai_track(description="Codegen - Relevant Warnings Step")
    def _invoke(self, **kwargs) -> None:
        self.logger.info("Executing Codegen - Relevant Warnings Step")
        self.context.event_manager.mark_running()
        diagnostics = []

        # 1. Read the PR.
        repo_client = self.context.get_repo_client(type=RepoClientType.READ)
        pr_files = repo_client.repo.get_pull(self.request.pr_id).get_files()
        pr_files = [
            PrFile(
                filename=file.filename,
                patch=file.patch,
                status=file.status,
                changes=file.changes,
                sha=file.sha,
            )
            for file in pr_files
            if file.patch
        ]
        diagnostics.append(
            {
                "component": "Relevant Warnings - Read PR",
                "pr_files": [pr_file.filename for pr_file in pr_files],
                "warnings": [warning.id for warning in self.request.warnings],
            }
        )

        # 2. Only consider warnings from lines changed in the PR.
        filter_warnings_component = FilterWarningsComponent(self.context)
        filter_warnings_request = FilterWarningsRequest(
            warnings=self.request.warnings, pr_files=pr_files
        )
        filter_warnings_output: FilterWarningsOutput = filter_warnings_component.invoke(
            filter_warnings_request
        )
        warning_and_pr_files = filter_warnings_output.warning_and_pr_files

        diagnostics.append(
            {
                "component": "Relevant Warnings - Filter Warnings Component",
                "filtered_warning_and_pr_files": [
                    [item.warning.id, item.pr_file.filename]
                    for item in filter_warnings_output.warning_and_pr_files
                ],
            }
        )
        if not warning_and_pr_files:  # exit early to avoid unnecessary issue-fetching.
            self.logger.info("No warnings to predict relevancy for.")
            self._complete_run(None, diagnostics)
            return

        # 3. Fetch issues related to the PR.
        fetch_issues_component = FetchIssuesComponent(self.context)
        fetch_issues_request = CodeFetchIssuesRequest(
            organization_id=self.request.organization_id, pr_files=pr_files
        )
        fetch_issues_output: CodeFetchIssuesOutput = fetch_issues_component.invoke(
            fetch_issues_request
        )
        # Clamp issue to max_num_issues_analyzed
        all_selected_issues = list(
            itertools.chain.from_iterable(fetch_issues_output.filename_to_issues.values())
        )
        all_selected_issues = all_selected_issues[: self.request.max_num_issues_analyzed]
        diagnostics.append(
            {
                "component": "Relevant Warnings - Fetch Issues Component",
                "all_selected_issues": [issue.id for issue in all_selected_issues],
            }
        )

        # 4. Limit the number of warning-issue associations we analyze to the top
        #    max_num_associations.
        association_component = AssociateWarningsWithIssuesComponent(self.context)
        associations_request = AssociateWarningsWithIssuesRequest(
            warning_and_pr_files=warning_and_pr_files,
            filename_to_issues=fetch_issues_output.filename_to_issues,
            max_num_associations=self.request.max_num_associations,
        )
        associations_output: AssociateWarningsWithIssuesOutput = association_component.invoke(
            associations_request
        )
        # Annotate the warnings with potential issues associated
        for association in associations_output.candidate_associations:
            assoc_warning, assoc_issue = association
            warning_from_list = next(
                (w for w in warning_and_pr_files if w.warning.id == assoc_warning.warning.id), None
            )
            if warning_from_list:
                if isinstance(warning_from_list.warning.potentially_related_issue_titles, list):
                    warning_from_list.warning.potentially_related_issue_titles.append(
                        assoc_issue.title
                    )
                else:
                    warning_from_list.warning.potentially_related_issue_titles = [assoc_issue.title]
        diagnostics.append(
            {
                "component": "Relevant Warnings - Associate Warnings With Issues Component",
                "candidate_associations": [
                    {
                        "warning_id": association[0].warning.id,
                        "pr_file": association[0].pr_file.filename,
                        "issue_id": association[1].id,
                    }
                    for association in associations_output.candidate_associations
                ],
                "potentially_related_issue_titles": [
                    {
                        "warning_id": warning_and_pr_file.warning.id,
                        "potentially_related_issue_titles": warning_and_pr_file.warning.potentially_related_issue_titles,
                        "pr_file": warning_and_pr_file.pr_file.filename,
                    }
                    for warning_and_pr_file in warning_and_pr_files
                ],
            }
        )

        # 5. Filter out unfixable issues b/c it doesn't make much sense to raise suggestions for issues you can't fix.
        are_issues_fixable_component = AreIssuesFixableComponent(self.context)
        are_fixable_output: CodeAreIssuesFixableOutput = are_issues_fixable_component.invoke(
            CodeAreIssuesFixableRequest(
                candidate_issues=all_selected_issues,
                max_num_issues_analyzed=self.request.max_num_issues_analyzed,
            )
        )
        fixable_issues = [
            issue
            for issue, is_fixable in zip(
                all_selected_issues,
                are_fixable_output.are_fixable,
                strict=True,
            )
            if is_fixable
        ]
        diagnostics.append(
            {
                "component": "Relevant Warnings - Are Issues Fixable Component",
                "fixable_issues": [issue.id for issue in fixable_issues],
            }
        )

        # 6. Suggest issues based on static analysis warnings and fixable issues.
        static_analysis_suggestions_component = StaticAnalysisSuggestionsComponent(self.context)
        static_analysis_suggestions_request = CodePredictStaticAnalysisSuggestionsRequest(
            warning_and_pr_files=warning_and_pr_files,
            fixable_issues=fixable_issues,
            pr_files=pr_files,
        )
        static_analysis_suggestions_output: CodePredictStaticAnalysisSuggestionsOutput = (
            static_analysis_suggestions_component.invoke(static_analysis_suggestions_request)
        )

        # 7. Save results.
        self._complete_run(static_analysis_suggestions_output, diagnostics)
