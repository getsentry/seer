import unittest

import numpy as np

from seer.anomaly_detection.detectors.window_size_selectors import SuSSWindowSizeSelector


class TestSuSSWindowSizeSelector(unittest.TestCase):

    def setUp(self):
        self.selector = SuSSWindowSizeSelector()

    def test_optimal_window_size_constant_series(self):
        ts = np.array([5.0] * 700)
        with self.assertRaises(Exception, msg="Search for optimal window failed."):
            self.selector.optimal_window_size(ts)

    def test_optimal_window_size_linear_series(self):
        ts = np.linspace(1, 100, 100)
        window_size = self.selector.optimal_window_size(ts)

        assert "Window size for linear series should be greater than the lower bound.", (
            window_size > 10
        )

    def test_optimal_window_size_short_series(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(Exception, msg="Search for optimal window failed."):
            self.selector.optimal_window_size(ts)
