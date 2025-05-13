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
from seer.automation.codegen.bug_prediction_component import BugPredictionFormatterComponent
from seer.automation.codegen.models import (
    BugPredictorFormatterInput,
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

    def _invoke(self, **kwargs) -> None:
        self.logger.info("Executing Codegen - Bug Prediction Step")
        self.context.event_manager.mark_running()

        """
        # 1. Read the PR.
        repo_client = self.context.get_repo_client(type=RepoClientType.READ)
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
        followups = temp_get_followups_inputs()
        llm_client = LlmClient()
        # end TODO

        # 4. Post-process each analysis—either into a presentable bug prediction if it's verified, else nothing.
        bug_prediction_formatter = BugPredictionFormatterComponent(self.context)
        formatted_predictions = bug_prediction_formatter.invoke(
            request=BugPredictorFormatterInput(followups=followups),
            llm_client=llm_client,
        )
        overwatch_results = []
        if formatted_predictions and hasattr(formatted_predictions, "formatted_predictions"):
            overwatch_results = [
                prediction.to_overwatch_result()
                for prediction in formatted_predictions.formatted_predictions
                if prediction.is_valid
            ]
        bug_predictions = CodePredictStaticAnalysisSuggestionsOutput(suggestions=overwatch_results)

        print(bug_predictions)

        # # 5. Save results.
        # self._complete_run(bug_predictions, diagnostics=None)


# TODO - remove this once we have a real flow
def temp_get_followups_inputs():
    from pathlib import Path

    # Go up to the main directory
    base_dir = Path(__file__).resolve().parents[4]  # Go up 4 levels from the current file
    followups_dir = base_dir / "tests" / "automation" / "codegen" / "fixtures" / "bug_prediction"
    followups = []

    if followups_dir.exists() and followups_dir.is_dir():
        for file_path in followups_dir.glob("*.txt"):
            with open(file_path, "r") as f:
                followups.append(f.read())

    return followups
