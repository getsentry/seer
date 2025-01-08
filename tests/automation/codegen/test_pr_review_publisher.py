import unittest
from unittest.mock import MagicMock, patch
from typing import List

from github.PullRequest import PullRequest

from seer.automation.codebase.models import GithubPrReviewComment
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.models import CodePrReviewOutput
from seer.automation.codegen.pr_review_publisher import PrReviewPublisher

class TestPrReviewPublisher(unittest.TestCase):
    def setUp(self):
        self.mock_repo_client = MagicMock(spec=RepoClient)
        self.mock_pr = MagicMock(spec=PullRequest)
        self.mock_pr.url = "https://api.github.com/repos/owner-name/repo-name/pulls/1"
        self.mock_pr.head.sha = "abcdef1234567890"

        self.publisher = PrReviewPublisher(
            repo_client=self.mock_repo_client,
            pr=self.mock_pr
        )

    def test_publish_no_changes_required(self):
        self.publisher.publish_no_changes_required()
        self.mock_repo_client.post_issue_comment.assert_called_once_with(
            self.mock_pr.url,
            "No changes requiring review at this time."
        )

    def test_publish_ack(self):
        self.publisher.publish_ack()
        self.mock_repo_client.post_issue_comment.assert_called_once_with(
            self.mock_pr.url,
            "On it! We are reviewing the PR and will provide feedback shortly."
        )

    def test_publish_generated_pr_review_no_comments(self):
        mock_pr_review_output = MagicMock(spec=CodePrReviewOutput)
        mock_pr_review_output.comments = []

        self.publisher.publish_generated_pr_review(mock_pr_review_output)
        self.mock_repo_client.post_issue_comment.assert_called_once_with(
            self.mock_pr.url,
            "No changes requiring review at this time."
        )

    def test_publish_generated_pr_review_with_comments(self):
        mock_comment = MagicMock(spec=GithubPrReviewComment)
        mock_comment.path = "file.py"
        mock_comment.line = 10
        mock_comment.body = "This is a review comment."

        mock_pr_review_output = MagicMock(spec=CodePrReviewOutput)
        mock_pr_review_output.comments = [mock_comment]

        with patch.object(
            PrReviewPublisher,
            '_format_comments',
            return_value=[mock_comment]
        ) as mock_format_comments:
            self.publisher.publish_generated_pr_review(mock_pr_review_output)

            self.mock_repo_client.post_pr_review_comment.assert_called_once_with(
                self.mock_pr.url,
                mock_comment
            )

            mock_format_comments.assert_called_once_with(
                self.mock_pr.head.sha,
                mock_pr_review_output
            )

    def test_publish_generated_pr_review_handles_exception(self): 
        mock_comment = MagicMock(spec=GithubPrReviewComment)
        mock_comment.path = "file.py"
        mock_comment.line = 10
        mock_comment.body = "This is a review comment."

        mock_pr_review_output = MagicMock(spec=CodePrReviewOutput)
        mock_pr_review_output.comments = [mock_comment]

        self.mock_repo_client.post_pr_review_comment.side_effect = ValueError("Invalid comment")

        with patch.object(
            PrReviewPublisher,
            '_format_comments',
            return_value=[mock_comment]
        ):
            self.publisher.publish_generated_pr_review(mock_pr_review_output)

            self.mock_repo_client.post_pr_review_comment.assert_called_once_with(
                self.mock_pr.url,
                mock_comment
            )

            self.mock_repo_client.post_issue_comment.assert_not_called()
