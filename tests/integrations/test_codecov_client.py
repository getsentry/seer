import unittest
from unittest.mock import patch, MagicMock
from integrations.codecov.codecov_client import CodecovClient
from seer.configuration import AppConfig


class MockConfig:
    CODECOV_SUPER_TOKEN = "test_token"


class TestCodecovClient(unittest.TestCase):
    @patch("integrations.codecov.codecov_client.requests.get")
    def test_fetch_coverage_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Mock coverage data"
        mock_get.return_value = mock_response

        result = CodecovClient.fetch_coverage("owner", "repo", "123", config=MockConfig())

        self.assertEqual(result, "Mock coverage data")
        mock_get.assert_called_once()

    @patch("integrations.codecov.codecov_client.requests.get")
    def test_fetch_coverage_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = CodecovClient.fetch_coverage("owner", "repo", "123", config=MockConfig())

        self.assertIsNone(result)
        mock_get.assert_called_once()

    @patch("integrations.codecov.codecov_client.requests.get")
    def test_fetch_test_results_for_commit_success_with_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Mock test results data"
        mock_response.json.return_value = {"count": 1}
        mock_get.return_value = mock_response

        result = CodecovClient.fetch_test_results_for_commit(
            "owner", "repo", "commit_sha", config=MockConfig()
        )

        self.assertEqual(result, "Mock test results data")
        mock_get.assert_called_once()

    @patch("integrations.codecov.codecov_client.requests.get")
    def test_fetch_test_results_for_commit_success_no_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 0}
        mock_get.return_value = mock_response

        result = CodecovClient.fetch_test_results_for_commit(
            "owner", "repo", "commit_sha", config=MockConfig()
        )

        self.assertIsNone(result)
        mock_get.assert_called_once()

    @patch("integrations.codecov.codecov_client.requests.get")
    def test_fetch_test_results_for_commit_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = CodecovClient.fetch_test_results_for_commit(
            "owner", "repo", "commit_sha", config=MockConfig()
        )

        self.assertIsNone(result)
        mock_get.assert_called_once()
