from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from celery_app.config import CeleryQueues
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.models import CodeUnitTestRequest
from seer.automation.codegen.step import CodegenStep
from seer.automation.codegen.unit_test_coding_component import UnitTestCodingComponent
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest


class UnittestStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def unittest_task(*args, request: dict[str, Any]):
    UnittestStep(request).invoke()


class UnittestStep(CodegenStep):
    """
    This class represents the unittest step in the codegen pipeline. It is responsible for
    generating unit tests based on the provided code changes in a pull request.
    """

    name = "UnittestStep"
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> UnittestStepRequest:
        return UnittestStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return unittest_task

    @observe(name="Codegen - Unittest Step")
    @ai_track(description="Codegen - Unittest Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - Unittest Step")

        repo_client = self.context.get_repo_client()
        pr = repo_client.repo.get_pull(self.request.pr_id)

        base_sha = pr.base.sha
        head_sha = pr.head.sha

        diff_content = repo_client.get_diff_content(base_sha, head_sha)

        unittest_output = UnitTestCodingComponent(self.context).invoke(
            CodeUnitTestRequest(
                diff=diff_content,
            )
        )

        if unittest_output:
            for file_change in unittest_output.diffs:
                self.append_file_change(file_change)

        self.send_unittest_result(unittest_output)

        # Add the next step in the pipeline here if needed
        # self.next(
        #     NextStep.get_signature(
        #         NextStepRequest(**self.step_request_fields)
        #     ),
        #     queue=CeleryQueues.DEFAULT,
        # )
