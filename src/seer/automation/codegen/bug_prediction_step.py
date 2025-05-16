import json
from typing import Any

import requests
from langfuse.decorators import observe

from celery_app.app import celery_app
from integrations.codecov.codecov_auth import get_codecov_auth_header
from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.models import PrFile
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.bug_prediction_component import BugPredictorFormatterComponent
from seer.automation.codegen.models import (
    BugPredictorFormatterInput,
    CodeBugPredictionsOutput,
    CodegenRelevantWarningsRequest,
    CodePredictStaticAnalysisSuggestionsOutput,
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
        bug_predictions: CodeBugPredictionsOutput | None,
        diagnostics: list | None,
        config: AppConfig = injected,
    ):
        if not self.request.should_post_to_overwatch:
            self.logger.info("Skipping posting relevant warnings results to Overwatch.")
            return

        request = {
            "run_id": self.context.run_id,
            "results": bug_predictions.bug_predictions,
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
        bug_predictions_output: CodeBugPredictionsOutput | None,
        diagnostics: list | None,
    ):
        try:
            self._post_results_to_overwatch(bug_predictions_output, diagnostics)
        except Exception:
            self.logger.exception("Error posting relevant warnings results to Overwatch")
            raise
        finally:
            self.context.event_manager.mark_completed_and_extend_static_analysis_suggestions(
                bug_predictions_output.bug_predictions if bug_predictions_output else []
            )

    @observe(name="Codegen - Bug Prediction Step")
    def _invoke(self, **kwargs) -> None:
        self.logger.info("Executing Codegen - Bug Prediction Step")
        self.context.event_manager.mark_running()

        """
        # 1. Read the PR.
        if self.request.repo.base_commit_sha is None:
            self.request.repo.base_commit_sha = self.request.commit_sha
        repo_client = self.context.get_repo_client(
            repo_name=self.request.repo.full_name, type=RepoClientType.READ
        )
        pr = repo_client.repo.get_pull(self.request.pr_id)
        pr_files = [
            PrFile(
                filename=file.filename,
                patch=file.patch,
                status=file.status,
                changes=file.changes,
                sha=file.sha,
                previous_filename=file.previous_filename,
            )
            for file in pr.get_files()
            if file.patch
        ]

        # 2. Filter files to ones that are most error prone.
        pr_files = (
            FilterFilesComponent(self.context)
            .invoke(FilterFilesRequest(pr_files=pr_files, pr_title=pr.title, pr_body=pr.body))
            .pr_files
        )
        # TODO: populate diagnostics

        # 3. Predict bugs, producing a thorough analysis of each hypothesis.
        bug_predictor_output = BugPredictorComponent(self.context).invoke(
            BugPredictorRequest(
                pr_files=pr_files,
                repo_full_name=self.request.repo.full_name,
                pr_title=pr.title,
                pr_body=pr.body,
            )
        )
        """

        # TODO - add here for local testing
        followups = temp_get_followups_inputs(2198)
        llm_client = LlmClient()
        # end TODO

        # 4. Post-process each analysis—either into a presentable bug prediction if it's verified, else nothing.
        bug_prediction_formatter = BugPredictorFormatterComponent(self.context)
        bug_predictions = bug_prediction_formatter.invoke(
            request=BugPredictorFormatterInput(followups=followups),
            llm_client=llm_client,
        )

        bug_predictions = CodeBugPredictionsOutput(bug_predictions=bug_predictions.bug_predictions)

        print(bug_predictions)

        # 5. Save results.
        self._complete_run(bug_predictions, diagnostics=None)


# TODO - remove this once we have a real flow
def temp_get_followups_inputs(pr_id: int):
    from pathlib import Path

    # Go up to the main directory
    base_dir = Path(__file__).resolve().parents[4]  # Go up 4 levels from the current file
    followups_dir = base_dir / "tests" / "automation" / "codegen" / "fixtures" / "bug_prediction"
    followups = []

    if followups_dir.exists() and followups_dir.is_dir():
        # Look for the specific PR ID directory
        pr_dir = followups_dir / str(pr_id)
        if pr_dir.exists() and pr_dir.is_dir():
            # Read all text files in the PR ID directory
            for file_path in sorted(pr_dir.glob("*.txt")):
                with open(file_path, "r") as f:
                    followups.append(f.read())

    return followups
