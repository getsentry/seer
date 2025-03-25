from asyncio.log import logger
from typing import Any

from github.PullRequestComment import PullRequestComment
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track
from sqlalchemy.dialects.postgresql import insert

from seer.automation.agent.embeddings import GoogleProviderEmbeddings 
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.step import CodegenStep
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes
from seer.db import DbReviewCommentEmbedding, Session
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

class PrClosedStepRequest(PipelineStepTaskRequest):
    pr_id: int
    repo_definition: RepoDefinition


class CommentAnalyzer:
    """
    Handles comment analysis logic
    """
    @inject
    def __init__(self, bot_id: str = None, config: AppConfig = injected):
        self.bot_id = bot_id or str(config.GITHUB_CODECOV_PR_REVIEW_APP_ID)

    def is_bot_comment(self, comment: PullRequestComment) -> bool:
        """Check if comment is authored by bot using app ID"""
        return str(comment.user.id) == self.bot_id

    def analyze_reactions(self, comment: PullRequestComment) -> tuple[bool, bool]:
        """
        Analyze reactions on a comment
        Returns: (is_good_pattern, is_bad_pattern)
        """
        reactions = comment.get_reactions()
        upvotes = sum(1 for r in reactions if r.content == "+1")
        downvotes = sum(1 for r in reactions if r.content == "-1")

        is_good_pattern = upvotes >= downvotes
        is_bad_pattern = downvotes > upvotes
        return is_good_pattern, is_bad_pattern


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
        pass

    def __init__(self, request: dict[str, Any], type: DbStateRunTypes):
        super().__init__(request, type)
        self.analyzer = CommentAnalyzer()

    def _process_comment(self, comment: PullRequestComment, pr):
        try:
            is_good_pattern, is_bad_pattern = self.analyzer.analyze_reactions(comment)

            logger.info(
                f"Processing bot comment id {comment.id} on PR {pr.url}: "
                f"good_pattern={is_good_pattern}, "
                f"bad_pattern={is_bad_pattern}"
            )

            try:
                model = GoogleProviderEmbeddings.model(
                    "text-embedding-005", task_type="SEMANTIC_SIMILARITY"
                )
                # encode() expects list[str], returns 2D array
                embedding = model.encode([comment.body])[0]
            except Exception as e:
                logger.warning(f"Failed to generate embeddings for comment {comment.id}: {e}")
                embedding = None

            with Session() as session:
                insert_stmt = insert(DbReviewCommentEmbedding).values(
                    provider="github",
                    owner=pr.base.repo.owner.login,
                    repo=pr.base.repo.name,
                    pr_id=pr.number,
                    body=comment.body,
                    is_good_pattern=is_good_pattern,
                    comment_metadata={
                        "url": comment.html_url,
                        "comment_id": comment.id,
                        "location": (
                            {"file_path": comment.path, "line_number": comment.position}
                            if hasattr(comment, "path")
                            else None
                        ),
                        "timestamps": {
                            "created_at": comment.created_at.isoformat(),
                            "updated_at": comment.updated_at.isoformat(),
                        },
                    },
                    embedding=embedding,
                )

                session.execute(
                    insert_stmt.on_conflict_do_nothing(
                        index_elements=["provider", "pr_id", "repo", "owner"]
                    )
                )
                session.commit()

        except Exception as e:
            self.logger.error(f"Error processing comment {comment.id} on PR {pr.url}: {e}")
            raise

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
