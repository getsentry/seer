import logging
from typing import List

from github.PullRequest import PullRequest

from seer.automation.codebase.models import GithubPrReviewComment
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.models import CodePrReviewOutput
from seer.automation.codegen.pr_review_utils import PrReviewUtils

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

        # Add PR description as a comment if available
        if pr_review.description:
            try:
                description_content = (
                    f"## PR Description\n\n"
                    f"{pr_review.description.purpose}\n\n"
                    f"<details>\n"
                    f"<summary><b>Click to see more</b></summary>\n\n"
                    f"### Key Technical Changes\n{pr_review.description.key_technical_changes}\n\n"
                    f"### Architecture Decisions\n{pr_review.description.architecture_decisions}\n\n"
                    f"### Dependencies and Interactions\n{pr_review.description.dependencies_and_interactions}\n\n"
                    f"### Risk Considerations\n{pr_review.description.risk_considerations}\n\n"
                    f"### Notable Implementation Details\n{pr_review.description.notable_implementation_details}\n"
                    f"</details>"
                )
                repo_client.post_issue_comment(pr_url, description_content)
            except ValueError as e:
                logger.warning(f"Failed to post PR description on PR {pr_url}: {e}")

        # handle send review comments one by one
        comments = self._format_comments(
            commit_id=self.pr.head.sha, pr_review=pr_review, owner=self.repo_client.repo_owner
        )
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
        commit_id: str, pr_review: CodePrReviewOutput, owner: str
    ) -> List[GithubPrReviewComment]:
        comments = []
        for comment in pr_review.comments:
            # TODO: rare case where start_line is greater than end line. Fix with a better prompt.
            if comment.line <= comment.start_line:
                continue
            if PrReviewUtils.is_positive_comment(comment.body, owner):
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
