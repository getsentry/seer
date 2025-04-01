import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sqlalchemy import select

from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.automation.codegen.pr_review_utils import SIMILARITY_THRESHOLD, PrReviewUtils
from seer.db import DbReviewCommentEmbedding, Session


class TestPrReviewUtils(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3]
        )  # Simplified embedding vector

        self.model_patcher = patch.object(
            GoogleProviderEmbeddings, "model", return_value=self.mock_model
        )
        self.mock_model_func = self.model_patcher.start()

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
        mock_comments = [
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=True),
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        self.session_mock.scalars.return_value.all.return_value = mock_comments
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

        self.session_mock.scalars.return_value.all.return_value = mock_comments

        result = PrReviewUtils.is_positive_comment("This is a negative comment", "test-owner")

        self.assertFalse(result)

    def test_is_positive_comment_with_not_enough_similar_comments(self):
        mock_comments = [
            MagicMock(is_good_pattern=False),
            MagicMock(is_good_pattern=False),
        ]

        self.session_mock.scalars.return_value.all.return_value = mock_comments
        result = PrReviewUtils.is_positive_comment(
            "Unique comment with few similar ones", "test-owner"
        )

        self.assertTrue(result)

    def test_is_positive_comment_with_exception(self):
        """Test behavior when an exception occurs during processing."""
        self.mock_model.encode.side_effect = Exception("Test exception")

        result = PrReviewUtils.is_positive_comment("Comment causing exception", "test-owner")

        self.assertFalse(result)
