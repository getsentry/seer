import unittest

import numpy as np

from seer.anomaly_detection.detectors.normalizers import MinMaxNormalizer

class TestNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = MinMaxNormalizer()

    def test_normalize_standard_array(self):
        normalizer = MinMaxNormalizer()
        array = np.array([1, 2, 3, 4, 5])
        expected = np.array([0, 0.25, 0.5, 0.75, 1])
        result = normalizer.normalize(array)
        np.testing.assert_equal(result, expected)

    def test_normalize_identical_values(self):
        array = np.array([3, 3, 3, 3])
        expected = np.array([np.nan, np.nan, np.nan, np.nan])
        result = self.normalizer.normalize(array)
        np.testing.assert_equal(result, expected)

    def test_empty_array(self):
        array = np.array([])
        self.assertRaises(ValueError, self.normalizer.normalize, array)

    def test_single_element_array(self):
        array = np.array([10])
        expected = np.array([np.nan])
        result = self.normalizer.normalize(array)
        np.testing.assert_equal(result, expected)
