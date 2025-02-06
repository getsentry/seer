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
from seer.automation.codegen.models import CodeRelevantWarningsOutput, CodeRelevantWarningsRequest
from seer.automation.codegen.relevant_warnings_component import RelevantWarningsComponent
from seer.automation.codegen.step import CodegenStep
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes


class RelevantWarningsStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def relevant_warnings_task(*args, request: dict[str, Any]):
    RelevantWarningsStep(request, DbStateRunTypes.RELEVANT_WARNINGS).invoke()


class RelevantWarningsStep(CodegenStep):
    """
    This class represents the Relevant Warnings step in the codegen pipeline. It is responsible for
    predicting which static analysis warnings in a pull request are relevant to a past Sentry issue.
    """

    name = "RelevantWarningsStep"
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

    @observe(name="Codegen - Relevant Warnings")
    @ai_track(description="Codegen - Relevant Warnings Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - Relevant Warnings Step")
        self.context.event_manager.mark_running()

        # TODO (important): ask codecov how we'll get this data.
        # client = self.context.warnings_and_issues_client()
        # candidate_warnings, candidate_issues = client.warnings_and_issues(self.request.pr_id)

        # TODO (important): make associations based on the locations of the warnings and issues in the codebase
        # https://github.com/codecov/bug-prediction-research/blob/main/src/scripts/make_associations.py
        # associations = self._associate_warnings_with_issue(candidate_warnings, candidate_issues)
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
        request = CodeRelevantWarningsRequest(candidate_associations=associations)

        matcher = RelevantWarningsComponent(self.context)
        relevant_warnings_output: CodeRelevantWarningsOutput = matcher.invoke(request)

        self.context.event_manager.mark_completed_and_extend_relevant_warning_results(
            relevant_warnings_output.relevant_warning_results
        )
