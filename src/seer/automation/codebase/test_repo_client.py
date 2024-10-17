import unittest
from unittest.mock import patch, MagicMock
from seer.automation.codebase.repo_client import RepoClient

class TestRepoClient(unittest.TestCase):
    def setUp(self):
        self.repo_client = RepoClient(token="test_token")

    @patch('seer.automation.codebase.repo_client.requests.post')
    def test_post_unit_test_reference_to_original_pr(self, mock_post):
        # Arrange
        original_pr_url = "https://github.com/sentry/sentry/pull/12345"
        unit_test_pr_url = "https://github.com/sentry/sentry/pull/67890"
        expected_url = "https://api.github.com/repos/sentry/sentry/issues/12345/comments"
        expected_comment = f"Sentry has generated a new [PR]({unit_test_pr_url}) with unit tests for this PR. View the new PR({unit_test_pr_url}) to review the changes."
        expected_params = {"body": expected_comment}
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"html_url": "https://github.com/sentry/sentry/pull/12345#issuecomment-1"}
        mock_post.return_value = mock_response

        # Act
        result = self.repo_client.post_unit_test_reference_to_original_pr(original_pr_url, unit_test_pr_url)

        # Assert
        mock_post.assert_called_once_with(
            expected_url,
            headers=self.repo_client._get_auth_headers(),
            json=expected_params
        )
        self.assertEqual(result, "https://github.com/sentry/sentry/pull/12345#issuecomment-1")

if __name__ == '__main__':
    unittest.main()