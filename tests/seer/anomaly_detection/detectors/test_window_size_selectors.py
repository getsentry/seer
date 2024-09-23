import unittest

import numpy as np

from seer.anomaly_detection.detectors.window_size_selectors import SuSSWindowSizeSelector
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestSuSSWindowSizeSelector(unittest.TestCase):

    def setUp(self):
        self.selector = SuSSWindowSizeSelector()

    def test_optimal_window_size_constant_series(self):
        ts = np.array([5.0] * 700)
        window_size = self.selector.optimal_window_size(ts)
        assert window_size == 3

    def test_optimal_window_size_linear_series(self):
        ts = np.linspace(1, 100, 100)
        window_size = self.selector.optimal_window_size(ts)
        assert "Window size for linear series should be greater than the lower bound.", (
            window_size > 10
        )

    def test_optimal_window_size_short_series(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = self.selector.optimal_window_size(ts)
        assert window_size == 3

    def test_optimal_window_size(self):

        actual_windows = []

        timeseries, _, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        for ts, window_size in zip(timeseries, window_sizes):
            window = self.selector.optimal_window_size(ts)
            actual_windows.append(window_size)

        # Check if window is within n% of period
        n = 0.6
        period = 24 * 4

        for window in actual_windows:
            assert window - (period * n) <= window <= window + (period * n)
