import unittest

from seer.anomaly_detection.detectors.window_size_selectors import SuSSWindowSizeSelector
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestSuSSWindowSizeSelector(unittest.TestCase):

    def setUp(self):
        self.selector = SuSSWindowSizeSelector()

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
            self.assertTrue(window - (period * n) <= window <= window + (period * n))
