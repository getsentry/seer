from typing import Any, List

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.models import GithubPrComment
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codebase.utils import format_pr_review
from seer.automation.codegen.models import CodePrReviewOutput, CodePrReviewRequest
from seer.automation.codegen.pr_review_coding_component import PrReviewCodingComponent
from seer.automation.codegen.step import CodegenStep
from seer.automation.models import FileChange, RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest

class PrReviewStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def pr_review_task(*args, request: dict[str, Any]):
    print("BEFORE INVOKE")
    PrReviewStep(request).invoke()


class PrReviewStep(CodegenStep):
    """
    This class represents the PR Review step in the codegen pipeline. It is responsible for
    generating pull request comments for provided code changes in a pull request.
    """

    name = "PrReviewStep"
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> PrReviewStepRequest:
        return PrReviewStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return pr_review_task

    @observe(name="Codegen - PR Review")
    @ai_track(description="Codegen - PR Review Step")
    def _invoke(self, **kwargs):
        print("HI 2")
        self.logger.info("Executing Codegen - PR Review Step")
        self.context.event_manager.mark_running()
        print("HI 3")

        repo_client = self.context.get_repo_client(type=RepoClientType.CODECOV_PR_REVIEW)
        pr = repo_client.repo.get_pull(self.request.pr_id)
        diff_content = repo_client.get_pr_diff_content(pr.url)
        print("BEFORE TRY")
        try:
            pr_review_output = PrReviewCodingComponent(self.context).invoke(
                CodePrReviewRequest(
                    diff=diff_content,
                ),
            )
            if pr_review_output:
                pr_review = format_pr_review(pr_review_output)
                try:
                    repo_client.post_pr_review(pr.url, pr_review)
                except ValueError:
                    return
            else:
                repo_client.post_pr_review_no_comments_required(pr.url)
                return

        except ValueError:
            return

        self.context.event_manager.mark_completed()

