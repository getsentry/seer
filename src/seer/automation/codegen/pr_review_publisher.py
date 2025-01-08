import logging
from typing import List

from github.PullRequest import PullRequest

from seer.automation.codebase.models import GithubPrReviewComment
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.models import (
    CodePrReviewOutput,
)

logger = logging.getLogger(__name__)


class PrReviewPublisher:
    """
    The PrReviewPublisher class publishes review comments to a pull request (PR)
    """

    def __init__(self, repo_client: RepoClient, pr: PullRequest):
        self.repo_client = repo_client
        self.pr = pr

    def publish_generated_pr_review(self, pr_review: CodePrReviewOutput) -> None:
        repo_client = self.repo_client
        pr_url = self.pr.url

        # handle if no comments
        if pr_review is None or not pr_review.comments:
            self.publish_no_changes_required()
            return

        # handle send review comments one by one
        comments = self._format_comments(self.pr.head.sha, pr_review)
        for comment in comments:
            try:
                repo_client.post_pr_review_comment(pr_url, comment)
            except ValueError as e:
                logger.warning(
                    f"Failed to post comment on PR {pr_url} for SHA {self.pr.head.sha}: {comment}. Error: {e}"
                )
                continue

    def publish_no_changes_required(self) -> None:
        self.repo_client.post_issue_comment(
            self.pr.url, "No changes requiring review at this time."
        )
        return

    def publish_ack(self) -> None:
        self.repo_client.post_issue_comment(
            self.pr.url, "On it! We are reviewing the PR and will provide feedback shortly."
        )
        return

    @staticmethod
    def _format_comments(
        commit_id: str, pr_review: CodePrReviewOutput
    ) -> List[GithubPrReviewComment]:
        comments = []
        for comment in pr_review.comments:
            comments.append(
                GithubPrReviewComment(
                    commit_id=commit_id,
                    side="RIGHT",
                    path=comment.path,
                    line=comment.line,
                    body=comment.body,
                    start_line=comment.start_line,
                )
            )

        return comments
