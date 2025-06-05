import json
from typing import Any

import requests
from langfuse.decorators import observe

from celery_app.app import celery_app
from integrations.codecov.codecov_auth import get_codecov_auth_header
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.models import PrFile, PullRequest
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.bug_prediction_component import (
    BugPredictorComponent,
    FilterFilesComponent,
    FormatterComponent,
)
from seer.automation.codegen.models import (
    BugPrediction,
    BugPredictorRequest,
    CodegenRelevantWarningsRequest,
    FilterFilesRequest,
    FormatterRequest,
)
from seer.automation.codegen.step import CodegenStep
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def bug_prediction_task(*args, request: dict[str, Any]):
    BugPredictionStep(request, DbStateRunTypes.BUG_PREDICTION).invoke()


class BugPredictionStepRequest(PipelineStepTaskRequest, CodegenRelevantWarningsRequest):
    pass


class BugPredictionStep(CodegenStep):
    name = "BugPredictionStep"
    request: BugPredictionStepRequest
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> BugPredictionStepRequest:
        return BugPredictionStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return bug_prediction_task

    @inject
    def _post_results_to_overwatch(
        self,
        bug_predictions: list[BugPrediction] | None,
        diagnostics: list | None,
        config: AppConfig = injected,
    ):
        if not self.request.should_post_to_overwatch:
            self.logger.info("Skipping posting relevant warnings results to Overwatch.")
            return

        bug_predictions_json = (
            [bp.model_dump() for bp in bug_predictions] if bug_predictions else []
        )

        request = {
            "run_id": self.context.run_id,
            "results": bug_predictions_json,
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
        bug_predictions: list[BugPrediction] | None,
        diagnostics: list | None,
    ):
        try:
            self._post_results_to_overwatch(bug_predictions, diagnostics)
        except Exception:
            self.logger.exception("Error posting relevant warnings results to Overwatch")
            raise
        finally:
            self.context.event_manager.mark_completed_and_extend_bug_predictions(
                bug_predictions if bug_predictions else []
            )

    @observe(name="Codegen - Bug Prediction Step")
    def _invoke(self, **kwargs) -> None:
        super()._invoke()

        self.logger.info("Executing Codegen - Bug Prediction Step")
        self.context.event_manager.mark_running()

        # 1. Read the PR.
        if self.request.repo.base_commit_sha is None:
            raise ValueError("The base_commit_sha must be set for bug prediction")

        repo_client = self.context.get_repo_client(
            repo_name=self.request.repo.full_name, type=RepoClientType.READ
        )

        pr = repo_client.repo.get_pull(self.request.pr_id)
        if pr.head.sha == self.request.repo.base_commit_sha:
            files = pr.get_files()
        else:  # Ensures signal from overwatch is consistent. Also used for evals.
            files = repo_client.repo.compare(
                base=pr.base.sha, head=self.request.repo.base_commit_sha
            ).files

        pr_files = [
            PrFile(
                filename=file.filename,
                patch=file.patch,
                status=file.status,
                changes=file.changes,
                sha=file.sha,
                previous_filename=file.previous_filename or file.filename,
                repo_full_name=self.request.repo.full_name,
            )
            for file in files
            if file.patch
        ]
        pull_request = PullRequest(files=pr_files, title=pr.title, description=pr.body)

        # 2. Filter files to ones that are most error prone.
        pull_request.files = (
            FilterFilesComponent(self.context)
            .invoke(FilterFilesRequest(pull_request=pull_request))
            .pr_files
        )
        # TODO: populate diagnostics

        # 3. Predict bugs, producing a thorough analysis of each hypothesis.
        bug_predictor_output = BugPredictorComponent(self.context).invoke(
            BugPredictorRequest(pull_request=pull_request)
        )

        # 4. Post-process the analyses into a list of presentable, verified bug predictions.
        bug_predictions = (
            FormatterComponent(self.context)
            .invoke(
                FormatterRequest(
                    located_followups=bug_predictor_output.get_located_followups(),
                ),
            )
            .bug_predictions
        )

        # 5. Save results.
        self._complete_run(bug_predictions, diagnostics=None)
