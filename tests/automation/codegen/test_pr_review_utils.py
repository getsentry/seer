import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sqlalchemy import select

from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.automation.codegen.pr_review_utils import SIMILARITY_THRESHOLD, PrReviewUtils
from seer.db import DbReviewCommentEmbedding, Session


class TestPrReviewUtils(unittest.TestCase):
    def setUp(self):
        # Setup mock for GoogleProviderEmbeddings
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3]
        )  # Simplified embedding vector

        # Setup patch for GoogleProviderEmbeddings.model
        self.model_patcher = patch.object(
            GoogleProviderEmbeddings, "model", return_value=self.mock_model
        )
        self.mock_model_func = self.model_patcher.start()

        # Setup Session mock
        self.session_mock = MagicMock()
        self.session_context_mock = MagicMock()
        self.session_context_mock.__enter__.return_value = self.session_mock
        self.session_patcher = patch.object(
            Session, "__call__", return_value=self.session_context_mock
        )
        self.mock_session = self.session_patcher.start()

    def tearDown(self):
        self.model_patcher.stop()
        self.session_patcher.stop()

    def test_is_positive_comment_with_majority_positive(self):
        """Test when there are more positive similar comments than negative ones."""
        # Setup mock comments with majority positive
        mock_comments = [
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        # Setup the session to return our mock comments
        self.session_mock.scalars.return_value.all.return_value = mock_comments

        # Call the method
        result = PrReviewUtils.is_positive_comment("This is a test comment", "test-owner")

        # Verify the result
        self.assertTrue(result)

        # Verify GoogleProviderEmbeddings.model was called with correct parameters
        self.mock_model_func.assert_called_once_with(
            "text-embedding-005", task_type="SEMANTIC_SIMILARITY"
        )

        # Verify encode was called with the comment content
        self.mock_model.encode.assert_called_once_with(["This is a test comment"])

        # Verify the session query was constructed correctly
        self.session_mock.scalars.assert_called_once()
        select_call = self.session_mock.scalars.call_args[0][0]
        self.assertIn("DbReviewCommentEmbedding", str(select_call))

    def test_is_positive_comment_with_majority_negative(self):
        """Test when there are more negative similar comments than positive ones."""
        # Setup mock comments with majority negative
        mock_comments = [
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        # Setup the session to return our mock comments
        self.session_mock.scalars.return_value.all.return_value = mock_comments

        # Call the method
        result = PrReviewUtils.is_positive_comment("This is a negative comment", "test-owner")

        # Verify the result
        self.assertFalse(result)

    def test_is_positive_comment_with_not_enough_similar_comments(self):
        """Test behavior when not enough similar comments are found."""
        # Return fewer comments than COMMENT_COMPARISON_LIMIT
        mock_comments = [
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        self.session_mock.scalars.return_value.all.return_value = mock_comments

        # Call the method - should default to True when not enough similar comments
        result = PrReviewUtils.is_positive_comment(
            "Unique comment with few similar ones", "test-owner"
        )

        # Verify the result
        self.assertTrue(result)

    def test_is_positive_comment_with_exception(self):
        """Test behavior when an exception occurs during processing."""
        # Make the model.encode method raise an exception
        self.mock_model.encode.side_effect = Exception("Test exception")

        # Call the method
        result = PrReviewUtils.is_positive_comment("Comment causing exception", "test-owner")

        # Should return False when an exception occurs
        self.assertFalse(result)

    def test_is_positive_comment_with_equal_split(self):
        """Test behavior when there's an equal split between positive and negative patterns."""
        mock_comments = [MagicMock(is_good_pattern=True), MagicMock(is_good_pattern=False)] * 2 + [
            MagicMock(is_good_pattern=True)
        ]
        self.session_mock.scalars.return_value.all.return_value = mock_comments

        # Call the method (3 positive out of 5 should be True)
        result = PrReviewUtils.is_positive_comment(
            "Comment with equal positive/negative patterns", "test-owner"
        )
        self.assertTrue(result)
