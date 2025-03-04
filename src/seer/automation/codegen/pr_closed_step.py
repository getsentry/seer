from asyncio.log import logger
from typing import Any
from github.PaginatedList import PaginatedList
from github.PullRequestComment import PullRequestComment

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


class PrClosedStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition


class CommentAnalyzer:
    """
    Handles comment analysis logic
    """
    def __init__(self, bot_username: str = "ai-review-test[bot]"):
        self.bot_username = bot_username

    def is_bot_comment(self, comment: PullRequestComment) -> bool:
        """Check if comment is authored by bot"""
        return comment.user.login == self.bot_username

    def analyze_reactions(self, comment: PullRequestComment) -> tuple[bool, bool]:
        """
        Analyze reactions on a comment
        Returns: (is_good_pattern, is_bad_pattern)
        """
        reactions = comment.get_reactions()
        upvotes = sum(1 for r in reactions if r.content == '+1')
        downvotes = sum(1 for r in reactions if r.content == '-1')
        
        is_good_pattern = upvotes >= downvotes 
        is_bad_pattern = downvotes > upvotes
        return is_good_pattern, is_bad_pattern


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def pr_closed_task(*args, request: dict[str, Any]):
    PrClosedStep(request, DbStateRunTypes.PR_CLOSED).invoke()


class PrClosedStep(CodegenStep):
    """
    This class represents the PR Closed step in the codegen pipeline. It is responsible for
    processing a closed or merged PR, including gathering and analyzing comment reactions.
    """

    name = "PrClosedStep"
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> PrClosedStepRequest:
        return PrClosedStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return pr_closed_task

    def __init__(self, request: dict[str, Any], type: DbStateRunTypes):
        super().__init__(request, type)
        self.analyzer = CommentAnalyzer()

    def _process_comment(self, comment: PullRequestComment, pr):
        """Helper method to process a single comment"""
        is_good_pattern, is_bad_pattern = self.analyzer.analyze_reactions(comment)
        
        logger.info(
            f"Processing bot comment id {comment.id} on PR {pr.url}: "
            f"good_pattern={is_good_pattern}, "
            f"bad_pattern={is_bad_pattern}"
        )

        # TODO: Save to DB:
        # You'll need three main steps:
        # Generate embeddings for each comment.
        # Compare embeddings to stored ones using cosine similarity.
        # Decide if a comment should pass or be blocked.


    @observe(name="Codegen - PR Closed")
    @ai_track(description="Codegen - PR Closed Step")
    def _invoke(self, **kwargs):
        self.logger.info("Executing Codegen - PR Closed Step")
        self.context.event_manager.mark_running()

        repo_client = self.context.get_repo_client(type=RepoClientType.CODECOV_PR_CLOSED)
        pr = repo_client.repo.get_pull(self.request.pr_id)

        try:
            review_comments = pr.get_review_comments()

            for comment in review_comments:
                if self.analyzer.is_bot_comment(comment):
                    self._process_comment(comment, pr)

            self.context.event_manager.mark_completed()
            
        except Exception as e:
            self.logger.error(f"Error processing closed PR {pr.url}: {e}")
            raise 