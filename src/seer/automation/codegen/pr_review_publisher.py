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
    def __init__(self, repo_client: RepoClient, pr: PullRequest):
        self.repo_client = repo_client
        self.pr = pr

    def publish(self, pr_review: CodePrReviewOutput) -> None:
        repo_client = self.repo_client
        pr_url = self.pr.url

        # handle if no comments
        if not pr_review or len(pr_review.comments) == 0:
            repo_client.post_pr_review_no_comments_required(pr_url)
            return

        # handle send review comments one by one
        comments = self._format_comments(self.pr.head.sha, pr_review)
        for comment in comments:
            try:
                repo_client.post_pr_review_comment(pr_url, comment)
            except ValueError:
                logger.warning(f"Failed to post comment: {comment}")
                continue

    @staticmethod
    def _format_comments(commit_id: str, pr_review: CodePrReviewOutput) -> List[GithubPrReviewComment]:
        comments = []
        for comment in pr_review.comments:
            c = GithubPrReviewComment(
                commit_id=commit_id,
                side="RIGHT",
                path=comment.path,
                line=comment.line,
                body=comment.body,
            )
            comments.append(c)

        return comments
