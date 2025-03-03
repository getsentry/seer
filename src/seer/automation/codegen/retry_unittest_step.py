from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)

from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.step import CodegenStep
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes
from seer.db import DbPrContextToUnitTestGenerationRunIdMapping


class RetryUnittestStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition
    codecov_status: dict


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def retry_unittest_task(*args, request: dict[str, Any]):
    RetryUnittestStep(request, DbStateRunTypes.UNIT_TESTS_RETRY).invoke()


class RetryUnittestStep(CodegenStep):
    """
    This class represents the retry unittest step in the codegen pipeline. It is responsible for
    updating generated unit tests based on the provided code changes and codecov TA and coverage information in a pull request.
    """

    name = "RetryUnittestStep"
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> RetryUnittestStepRequest:
        return RetryUnittestStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return retry_unittest_task

    @observe(name="Codegen - Retry Unittest Step")
    @ai_track(description="Codegen - Retry Unittest Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - Retry Unittest Step")
        self.context.event_manager.mark_running()
        repo_client = self.context.get_repo_client(
            type=RepoClientType.CODECOV_PR_REVIEW
        )  # Codecov-ai GH app
        pr = repo_client.repo.get_pull(self.request.pr_id)
        codecov_status = self.request.codecov_status["check_run"]["conclusion"]

        if codecov_status == "success":
            saved_memory = DbPrContextToUnitTestGenerationRunIdMapping.objects.filter(
                owner=self.request.owner,
                repo=self.request.repo_definition.name,
                pr_id=self.request.pr_id,
            ).first()

            repo_client.post_unit_test_reference_to_original_pr(
                saved_memory.original_pr_url, pr.html_url
            )

        else:
            past_run = DbPrContextToUnitTestGenerationRunIdMapping.objects.filter(
                owner=self.request.owner,
                repo=self.request.repo_definition.name,
                pr_id=self.request.pr_id,
            ).first()
            if not past_run:
                return
            if past_run.iterations == 3:
                # TODO: Fetch the "best" run and update the PR
                return
            else:
                # TODO: Retry test generation
                pass
        self.context.event_manager.mark_completed()
