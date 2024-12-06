import unittest
from unittest.mock import MagicMock, patch
from typing import List

from seer.automation.codebase.models import GithubPrReviewComment
from seer.automation.codegen.pr_review_publisher import PrReviewPublisher


# Mock dependencies
class RepoClient:
    def post_pr_review_no_comments_required(self, pr_url: str):
        pass

    def post_pr_review_comment(self, pr_url: str, comment: GithubPrReviewComment):
        pass


class PullRequest:
    def __init__(self, url: str):
        self.url = url


class GithubPrReviewComment:
    def __init__(self, path: str, line: int, body: str, start_line: int):
        self.path = path
        self.line = line
        self.body = body
        self.start_line = start_line


class CodePrReviewOutput:
    class Comment:
        def __init__(self, path: str, line: int, body: str, start_line: int):
            self.path = path
            self.line = line
            self.body = body
            self.start_line = start_line

    def __init__(self, comments: List[Comment]):
        self.comments = comments


# Test Class
class TestPrReviewPublisher(unittest.TestCase):
    def setUp(self):
        self.repo_client = MagicMock(spec=RepoClient)
        self.pr = PullRequest(url="https://github.com/test/repo/pull/123")
        self.publisher = PrReviewPublisher(self.repo_client, self.pr)

    def test_publish_no_comments(self):
        pr_review = CodePrReviewOutput(comments=[])
        self.publisher.publish(pr_review)

        self.repo_client.post_pr_review_no_comments_required.assert_called_once_with(self.pr.url)
        self.repo_client.post_pr_review_comment.assert_not_called()

    def test_publish_with_comments(self):
        pr_review = CodePrReviewOutput(
            comments=[
                CodePrReviewOutput.Comment(path="file1.py", line=10, comment="Fix this", start_line=5),
                CodePrReviewOutput.Comment(path="file2.py", line=20, comment="Improve this", start_line=15),
            ]
        )
        self.publisher.publish(pr_review)

        self.repo_client.post_pr_review_no_comments_required.assert_not_called()
        self.repo_client.post_pr_review_comment.assert_any_call(
            self.pr.url,
            GithubPrReviewComment(path="file1.py", line=10, body="Fix this", start_line=5),
        )
        self.repo_client.post_pr_review_comment.assert_any_call(
            self.pr.url,
            GithubPrReviewComment(path="file2.py", line=20, body="Improve this", start_line=15),
        )
        self.assertEqual(self.repo_client.post_pr_review_comment.call_count, 2)

    @patch("logger.warning")
    def test_publish_with_comment_post_failure(self, mock_logger_warning):
        pr_review = CodePrReviewOutput(
            comments=[
                CodePrReviewOutput.Comment(path="file1.py", line=10, comment="Fix this", start_line=5),
            ]
        )
        self.repo_client.post_pr_review_comment.side_effect = [ValueError("Failed to post")]

        self.publisher.publish(pr_review)

        self.repo_client.post_pr_review_no_comments_required.assert_not_called()
        self.repo_client.post_pr_review_comment.assert_called_once()
        mock_logger_warning.assert_called_once_with(
            "Failed to post comment: GithubPrReviewComment(path='file1.py', line=10, body='Fix this', start_line=5)"
        )

    def test_format_comments(self):
        pr_review = CodePrReviewOutput(
            comments=[
                CodePrReviewOutput.Comment(path="file1.py", line=10, comment="Fix this", start_line=5),
                CodePrReviewOutput.Comment(path="file2.py", line=20, comment="Improve this", start_line=15),
            ]
        )
        formatted_comments = PrReviewPublisher._format_comments(pr_review)

        self.assertEqual(len(formatted_comments), 2)
        self.assertEqual(formatted_comments[0].path, "file1.py")
        self.assertEqual(formatted_comments[0].line, 10)
        self.assertEqual(formatted_comments[0].body, "Fix this")
        self.assertEqual(formatted_comments[0].start_line, 5)
        self.assertEqual(formatted_comments[1].path, "file2.py")
        self.assertEqual(formatted_comments[1].line, 20)
        self.assertEqual(formatted_comments[1].body, "Improve this")
        self.assertEqual(formatted_comments[1].start_line, 15)
