import unittest
from unittest.mock import MagicMock

import numpy as np

from seer.anomaly_detection.detectors import MPConfig, Normalizer
from seer.anomaly_detection.detectors.mp_utils import MPUtils


class TestMPUtils(unittest.TestCase):

    def setUp(self):
        self.utils = MPUtils()

        # Mock MPConfig and Normalizer
        self.mock_mp_config = MagicMock(spec=MPConfig)
        self.mock_normalizer = MagicMock(spec=Normalizer)

    def test_no_padding_no_normalization(self):
        mp = np.array([[1.0], [2.0], [3.0], [4.0]])
        self.mock_mp_config.normalize_mp = False

        result = self.utils.get_mp_dist_from_mp(
            mp, pad_to_len=None, mp_config=self.mock_mp_config, normalizer=None
        )

        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_no_padding_with_normalization(self):
        mp = np.array([[1.0], [2.0], [3.0], [4.0]])
        self.mock_mp_config.normalize_mp = True
        self.mock_normalizer.normalize.return_value = np.array([0.0, 0.33, 0.67, 1.0])

        result = self.utils.get_mp_dist_from_mp(
            mp, pad_to_len=None, mp_config=self.mock_mp_config, normalizer=self.mock_normalizer
        )

        expected = np.array([0.0, 0.33, 0.67, 1.0], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_padding_no_normalization(self):
        mp = np.array([[1.0], [2.0]])
        self.mock_mp_config.normalize_mp = False

        result = self.utils.get_mp_dist_from_mp(
            mp, pad_to_len=5, mp_config=self.mock_mp_config, normalizer=None
        )

        expected = np.array([np.nan, np.nan, np.nan, 1.0, 2.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_padding_with_normalization(self):
        mp = np.array([[1.0], [2.0]])
        self.mock_mp_config.normalize_mp = True
        self.mock_normalizer.normalize.return_value = np.array([0.0, 1.0])

        result = self.utils.get_mp_dist_from_mp(
            mp, pad_to_len=5, mp_config=self.mock_mp_config, normalizer=self.mock_normalizer
        )

        expected = np.array([np.nan, np.nan, np.nan, 0.0, 1.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_no_normalizer_with_normalization(self):
        mp = np.array([[1.0], [2.0], [3.0], [4.0]])
        self.mock_mp_config.normalize_mp = True

        with self.assertRaises(Exception) as context:
            self.utils.get_mp_dist_from_mp(
                mp, pad_to_len=None, mp_config=self.mock_mp_config, normalizer=None
            )

        assert "Need normalizer to normalize MP" in str(context.exception)

    def test_incorrect_padding(self):

        mp = np.array([[1.0], [2.0], [3.0], [4.0]])
        self.mock_mp_config.normalize_mp = False

        with self.assertRaises(Exception) as context:
            self.utils.get_mp_dist_from_mp(
                mp, pad_to_len=1, mp_config=self.mock_mp_config, normalizer=None
            )
            assert (
                context.msg == "Requested length should be greater than or equal to current mp_dist"
            )
