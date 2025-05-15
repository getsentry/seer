from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.models import CodePrReviewRequest, PrAdditionalContextRequest, PrFile
from seer.automation.codegen.pr_additional_context_component import PrAdditionalContextComponent
from seer.automation.codegen.pr_review_coding_component import PrReviewCodingComponent
from seer.automation.codegen.pr_review_publisher import PrReviewPublisher
from seer.automation.codegen.relevant_warnings_component import (
    CodeFetchIssuesOutput,
    CodeFetchIssuesRequest,
    FetchIssuesComponent,
)
from seer.automation.codegen.step import CodegenStep
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes


class PrReviewStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def pr_review_task(*args, request: dict[str, Any]):
    PrReviewStep(request, DbStateRunTypes.PR_REVIEW).invoke()


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
        self.logger.info("Executing Codegen - PR Review Step")
        self.context.event_manager.mark_running()

        repo_client = self.context.get_repo_client(type=RepoClientType.CODECOV_PR_REVIEW)
        pr = repo_client.repo.get_pull(self.request.pr_id)
        diff_content = repo_client.get_pr_diff_content(pr.url)
        pr_files = pr.get_files()
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

        generator = PrReviewCodingComponent(self.context)
        publisher = PrReviewPublisher(repo_client=repo_client, pr=pr)

        # 1. Publish initial comment on PR that we are working on it
        try:
            publisher.publish_ack()
        except Exception as e:
            self.logger.warning(f"Error publishing ack for {pr.url}: {e}")
            # proceed even if ack fails
            pass

        # 2. Fetch additional context for PR review
        try:
            fetch_issues_component = FetchIssuesComponent(self.context)
            fetch_issues_request = CodeFetchIssuesRequest(
                organization_id=self.request.organization_id, pr_files=pr_files
            )
            fetch_issues_output: CodeFetchIssuesOutput = fetch_issues_component.invoke(
                fetch_issues_request
            )
            additional_context_component = PrAdditionalContextComponent(self.context)
            additional_context = additional_context_component.invoke(
                request=PrAdditionalContextRequest(
                    pr_files=pr_files,
                    filename_to_issues=fetch_issues_output.filename_to_issues,
                )
            )
            additional_context_prompt = additional_context.to_llm_prompt()
        except Exception as e:
            self.logger.error(f"Error fetching additional context for {pr.url}: {e}")
            # proceed even if additional context fails
            additional_context_prompt = ""
            pass

        # 3. Generate PR review
        try:
            generated_pr_review = generator.invoke(
                CodePrReviewRequest(
                    diff=diff_content,
                    additional_context=additional_context_prompt,
                ),
            )
        except ValueError as e:
            self.logger.error(f"Error generating pr review for {pr.url}: {e}")
            # send "no changes required" upon generation error including if output
            # does not conform to expected format
            publisher.publish_no_changes_required()
            return

        # 4. Publish generated PR review
        try:
            publisher.publish_generated_pr_review(pr_review=generated_pr_review),
        except ValueError as e:
            self.logger.error(f"Error publishing pr review for {pr.url}: {e}")
            return

        self.context.event_manager.mark_completed()
