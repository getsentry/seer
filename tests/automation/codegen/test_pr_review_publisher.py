import unittest
from unittest.mock import MagicMock, patch

from github.PullRequest import PullRequest

from seer.automation.codebase.models import GithubPrReviewComment
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.models import CodePrReviewOutput
from seer.automation.codegen.pr_review_publisher import PrReviewPublisher
from seer.automation.codegen.pr_review_utils import PrReviewUtils


class TestPrReviewPublisher(unittest.TestCase):
    def setUp(self):
        self.mock_repo_client = MagicMock(spec=RepoClient)
        self.mock_repo_client.repo = MagicMock()
        self.mock_repo_client.repo.owner = MagicMock()
        self.mock_repo_client.repo.owner.login = "owner-name"
        self.mock_repo_client.repo_owner = "owner-name"
        self.mock_pr = MagicMock(spec=PullRequest)
        self.mock_pr.url = "https://api.github.com/repos/owner-name/repo-name/pulls/1"
        self.mock_pr.head.sha = "abcdef1234567890"

        self.publisher = PrReviewPublisher(repo_client=self.mock_repo_client, pr=self.mock_pr)

    def test_publish_no_changes_required(self):
        self.publisher.publish_no_changes_required()
        self.mock_repo_client.post_issue_comment.assert_called_once_with(
            self.mock_pr.url, "No changes requiring review at this time."
        )

    def test_publish_ack(self):
        self.publisher.publish_ack()
        self.mock_repo_client.post_issue_comment.assert_called_once_with(
            self.mock_pr.url, "On it! We are reviewing the PR and will provide feedback shortly."
        )

    def test_publish_generated_pr_review_no_comments(self):
        mock_pr_review_output = MagicMock(spec=CodePrReviewOutput)
        mock_pr_review_output.comments = []
        mock_pr_review_output.description = None

        self.publisher.publish_generated_pr_review(mock_pr_review_output)
        self.mock_repo_client.post_issue_comment.assert_called_once_with(
            self.mock_pr.url, "No changes requiring review at this time."
        )

    def test_publish_generated_pr_review_with_comments(self):
        mock_comment = MagicMock(spec=GithubPrReviewComment)
        mock_comment.path = "file.py"
        mock_comment.line = 10
        mock_comment.body = "This is a review comment."

        mock_pr_review_output = MagicMock(spec=CodePrReviewOutput)
        mock_pr_review_output.comments = [mock_comment]
        mock_pr_review_output.description = None

        with patch.object(
            PrReviewPublisher, "_format_comments", return_value=[mock_comment]
        ) as mock_format_comments:
            self.publisher.publish_generated_pr_review(mock_pr_review_output)

            self.mock_repo_client.post_pr_review_comment.assert_called_once_with(
                self.mock_pr.url, mock_comment
            )

            mock_format_comments.assert_called_once_with(
                commit_id=self.mock_pr.head.sha,
                pr_review=mock_pr_review_output,
                owner=self.mock_repo_client.repo_owner,
            )

    def test_publish_generated_pr_review_handles_exception(self):
        mock_comment = MagicMock(spec=GithubPrReviewComment)
        mock_comment.path = "file.py"
        mock_comment.line = 10
        mock_comment.body = "This is a review comment."

        mock_pr_review_output = MagicMock(spec=CodePrReviewOutput)
        mock_pr_review_output.comments = [mock_comment]
        mock_pr_review_output.description = None

        self.mock_repo_client.post_pr_review_comment.side_effect = ValueError("Invalid comment")

        with patch.object(PrReviewPublisher, "_format_comments", return_value=[mock_comment]):
            self.publisher.publish_generated_pr_review(mock_pr_review_output)

            self.mock_repo_client.post_pr_review_comment.assert_called_once_with(
                self.mock_pr.url, mock_comment
            )

            self.mock_repo_client.post_issue_comment.assert_not_called()

    def test_format_comments_filtering(self):
        # Create a CodePrReviewOutput with comments
        review_comment1 = MagicMock()
        review_comment1.path = "file1.py"
        review_comment1.line = 10
        review_comment1.start_line = 5
        review_comment1.body = "Good comment"

        review_comment2 = MagicMock()
        review_comment2.path = "file2.py"
        review_comment2.line = 20
        review_comment2.start_line = 15
        review_comment2.body = "Bad comment"

        review_comment3 = MagicMock()
        review_comment3.path = "file3.py"
        review_comment3.line = 5  # Line <= start_line, should be skipped
        review_comment3.start_line = 5
        review_comment3.body = "Another comment"

        pr_review = MagicMock(spec=CodePrReviewOutput)
        pr_review.comments = [review_comment1, review_comment2, review_comment3]
        pr_review.description = None

        commit_id = "test-commit-id"
        owner = "owner-name"

        # Mock PrReviewUtils.is_positive_comment to return True for comment1, False for comment2
        with patch.object(PrReviewUtils, "is_positive_comment") as mock_is_positive:
            mock_is_positive.side_effect = (
                lambda comment, o: comment == "Good comment" and o == owner
            )

            # Call _format_comments method
            result = PrReviewPublisher._format_comments(
                commit_id=commit_id, pr_review=pr_review, owner=owner
            )

            # Verify that only the positive comment was included
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["path"], "file1.py")
            self.assertEqual(result[0]["body"], "Good comment")
            self.assertEqual(result[0]["commit_id"], commit_id)

            # Verify is_positive_comment was called with correct parameters for review_comment1 and review_comment2
            # (not called for review_comment3 since it's filtered by line <= start_line)
            mock_is_positive.assert_any_call("Good comment", owner)
            mock_is_positive.assert_any_call("Bad comment", owner)
            self.assertEqual(mock_is_positive.call_count, 2)

    def test_publish_generated_pr_review_with_structured_description(self):
        # Create a mock PR review with a structured description
        mock_pr_description = MagicMock()
        mock_pr_description.purpose = "Test purpose"
        mock_pr_description.key_technical_changes = "Test technical changes"
        mock_pr_description.architecture_decisions = "Test architecture decisions"
        mock_pr_description.dependencies_and_interactions = "Test dependencies"
        mock_pr_description.risk_considerations = "Test risks"
        mock_pr_description.notable_implementation_details = "Test implementation details"

        mock_comment = MagicMock()
        mock_comment.path = "file1.py"
        mock_comment.line = 10
        mock_comment.start_line = 5
        mock_comment.body = "Test comment"

        mock_pr_review = MagicMock(spec=CodePrReviewOutput)
        mock_pr_review.comments = [mock_comment]
        mock_pr_review.description = mock_pr_description

        # Mock the necessary objects
        mock_repo_client = self.mock_repo_client
        mock_pr = self.mock_pr
        mock_pr.head.sha = "test-sha"

        # Create publisher instance
        publisher = PrReviewPublisher(mock_repo_client, mock_pr)

        # Mock the _format_comments method to return a predefined result
        with patch.object(PrReviewPublisher, "_format_comments") as mock_format_comments:
            mock_format_comments.return_value = [{"dummy": "comment"}]

            # Call the method under test
            publisher.publish_generated_pr_review(mock_pr_review)

            # Verify the PR description was posted with the correct format
            expected_description_content = (
                f"## PR Description\n\n"
                f"{mock_pr_description.purpose}\n\n"
                f"<details>\n"
                f"<summary><b>Click to see more</b></summary>\n\n"
                f"### Key Technical Changes\n{mock_pr_description.key_technical_changes}\n\n"
                f"### Architecture Decisions\n{mock_pr_description.architecture_decisions}\n\n"
                f"### Dependencies and Interactions\n{mock_pr_description.dependencies_and_interactions}\n\n"
                f"### Risk Considerations\n{mock_pr_description.risk_considerations}\n\n"
                f"### Notable Implementation Details\n{mock_pr_description.notable_implementation_details}\n"
                f"</details>"
            )

            mock_repo_client.post_issue_comment.assert_called_with(
                mock_pr.url, expected_description_content
            )
