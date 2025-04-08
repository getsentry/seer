import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.automation.codegen.pr_review_utils import PrReviewUtils


class TestPrReviewUtils(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.full((768,), 0.1)
        self.model_patcher = patch.object(
            GoogleProviderEmbeddings, "model", return_value=self.mock_model
        )
        self.model_patcher.start()

        # Patch the Session in the module where it is used
        self.session_patcher = patch("seer.automation.codegen.pr_review_utils.Session")
        self.mock_session_class = self.session_patcher.start()
        # Create a mock context manager for Session
        self.session_context_mock = MagicMock()
        self.session_context_mock.__enter__.return_value = self.session_context_mock
        self.session_context_mock.__exit__.return_value = None
        self.mock_session_class.return_value = self.session_context_mock

        # Set up the query chain
        self.query_mock = MagicMock()
        self.query_mock.where.return_value = self.query_mock
        self.query_mock.order_by.return_value = self.query_mock
        self.query_mock.limit.return_value = self.query_mock
        self.session_context_mock.query.return_value = self.query_mock

    def tearDown(self):
        self.model_patcher.stop()
        self.session_patcher.stop()

    def test_is_positive_comment_with_majority_positive(self):
        mock_comments = [
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        self.query_mock.all.return_value = mock_comments
        result = PrReviewUtils.is_positive_comment("This is a test comment", "test-owner")
        self.assertTrue(result)

    def test_is_positive_comment_with_majority_negative(self):
        # Setup mock comments with majority negative
        mock_comments = [
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        self.query_mock.all.return_value = mock_comments

        result = PrReviewUtils.is_positive_comment("This is a negative comment", "test-owner")

        self.assertFalse(result)

    def test_is_positive_comment_with_not_enough_similar_comments(self):
        mock_comments = [
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        self.query_mock.all.return_value = mock_comments
        result = PrReviewUtils.is_positive_comment(
            "Unique comment with few similar ones", "test-owner"
        )

        self.assertTrue(result)

    def test_is_positive_comment_with_exception(self):
        """Test behavior when an exception occurs during processing."""
        self.mock_model.encode.side_effect = Exception("Test exception")

        result = PrReviewUtils.is_positive_comment("Comment causing exception", "test-owner")

        self.assertTrue(result)
