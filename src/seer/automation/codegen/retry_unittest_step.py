from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)

# from seer.automation.codegen.retry_unit_test_coding_component import RetryUnitTestCodingComponent
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.models import CodeUnitTestRequest
from seer.automation.codegen.retry_unittest_coding_component import RetryUnitTestCodingComponent
from seer.automation.codegen.step import CodegenStep
from seer.automation.codegen.unit_test_github_pr_creator import GeneratedTestsPullRequestCreator
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes
from seer.automation.utils import determine_mapped_unit_test_run_id


class RetryUnittestStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition
    # codecov_status: dict


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
        x = retry_unittest_task
        return x

    @observe(name="Codegen - Retry Unittest Step")
    @ai_track(description="Codegen - Retry Unittest Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - Retry Unittest Step")
        self.context.event_manager.mark_running()
        # TODO: IF STATUS CHECK HAS PASSED OR WE HAVE MORE THAN 3 COMMITS, SKIP UNIT TEST GENERATION:

        repo_client = self.context.get_repo_client(type=RepoClientType.CODECOV_UNIT_TEST)
        pr = repo_client.repo.get_pull(self.request.pr_id)
        diff_content = repo_client.get_pr_diff_content(pr.url)

        latest_commit_sha = repo_client.get_pr_head_sha(pr.url)

        codecov_client_params = {
            "repo_name": self.request.repo_definition.name,
            "pullid": self.request.pr_id,
            "owner_username": self.request.repo_definition.owner,
            "head_sha": latest_commit_sha,
        }
        try:
            unittest_output = RetryUnitTestCodingComponent(self.context).invoke(
                CodeUnitTestRequest(
                    diff=diff_content,
                    codecov_client_params=codecov_client_params,
                ),
                generated_run_id=determine_mapped_unit_test_run_id(
                    owner=self.request.repo_definition.owner,
                    repo_name=self.request.repo_definition.name,
                    pr_id=self.request.pr_id,
                ),
            )

            if unittest_output:
                for file_change in unittest_output.diffs:
                    self.context.event_manager.append_file_change(file_change)
                generator = GeneratedTestsPullRequestCreator(unittest_output.diffs, pr, repo_client)
                generator.create_github_pull_request()
            else:
                repo_client.post_unit_test_not_generated_message_to_original_pr(pr.html_url)
                return

        except ValueError:
            repo_client.post_unit_test_not_generated_message_to_original_pr(pr.html_url)
        self.context.event_manager.mark_completed()
