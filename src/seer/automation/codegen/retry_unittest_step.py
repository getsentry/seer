from typing import Any, Optional

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.models import CodeUnitTestRequest
from seer.automation.codegen.retry_unit_test_github_pr_creator import RetryUnitTestGithubPrUpdater
from seer.automation.codegen.retry_unittest_coding_component import RetryUnitTestCodingComponent
from seer.automation.codegen.step import CodegenStep
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes


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
    Retry unittest step in the codegen pipeline.

    Responsible for updating generated unit tests based on code changes and codecov
    TA and coverage information in a pull request.
    """

    name = "RetryUnittestStep"
    request: RetryUnittestStepRequest
    max_retries = 2
    MAX_ITERATIONS = 4

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

        try:
            repo_client = self.context.get_repo_client(
                repo_name=self.request.repo_definition.full_name,
                type=RepoClientType.CODECOV_PR_REVIEW,
            )  # Codecov AI GH app
            pr = repo_client.repo.get_pull(self.request.pr_id)
            previous_run_context = self._get_previous_run_context()

            codecov_status = self.request.codecov_status["conclusion"]
            if codecov_status == "success":
                self._handle_successful_pr_status(repo_client, previous_run_context, pr)
            else:
                self._handle_failed_pr_status(repo_client, previous_run_context, pr)

        except Exception as e:
            self.logger.exception(f"Error in RetryUnittestStep: {str(e)}")
            raise
        finally:
            self.context.event_manager.mark_completed()

    def _get_previous_run_context(self) -> Any:
        previous_run_context = self.context.get_previous_run_context(
            self.request.repo_definition.owner,
            self.request.repo_definition.name,
            self.request.pr_id,
        )

        if not previous_run_context:
            raise RuntimeError("Unable to find PR context to retry unit tests")

        return previous_run_context

    def _handle_successful_pr_status(self, repo_client, previous_run_context, pr) -> None:
        repo_client.post_unit_test_reference_to_original_pr_codecov_app(
            previous_run_context.original_pr_url, pr.html_url
        )

    def _handle_failed_pr_status(self, repo_client, previous_run_context, pr) -> None:
        if previous_run_context.iterations >= self.MAX_ITERATIONS:
            self.logger.info("Maximum iterations reached, skipping retry")
            # TODO: Fetch the "best" run and update the PR
            return

        self.logger.info(f"Retrying unit tests (iteration {previous_run_context.iterations + 1})")
        unittest_output = self._generate_unit_tests(repo_client, pr, previous_run_context)

        if unittest_output:
            for file_change in unittest_output.diffs:
                self.context.event_manager.append_file_change(file_change)
            generator = RetryUnitTestGithubPrUpdater(
                file_changes_payload=unittest_output.diffs,
                pr=pr,
                repo_client=repo_client,
                previous_context=previous_run_context,
            )
            generator.update_github_pull_request()

        else:
            self.logger.info("No unit tests generated, posting message to original PR")
            repo_client.post_unit_test_reference_to_original_pr_codecov_app(
                previous_run_context.original_pr_url, pr.html_url
            )

    def _generate_unit_tests(self, repo_client, pr, previous_run_context) -> Optional[Any]:
        try:
            diff_content = repo_client.get_pr_diff_content(pr.url)
            latest_commit_sha = repo_client.get_pr_head_sha(pr.url)

            codecov_client_params = {
                "repo_name": self.request.repo_definition.name,
                "pullid": self.request.pr_id,
                "owner_username": self.request.repo_definition.owner,
                "head_sha": latest_commit_sha,
            }

            return RetryUnitTestCodingComponent(self.context).invoke(
                CodeUnitTestRequest(
                    diff=diff_content,
                    codecov_client_params=codecov_client_params,
                ),
                previous_run_context=previous_run_context,
            )
        except Exception as e:
            self.logger.exception(f"Error generating unit tests: {str(e)}")
            return None
