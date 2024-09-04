import unittest

from seer.anomaly_detection.detectors.window_size_selectors import SuSSWindowSizeSelector


class TestSuSSWindowSizeSelector(unittest.TestCase):

    def setUp(self):
        self.selector = SuSSWindowSizeSelector()

    def test_optimal_window_size(self):
        # TODO: Import time series
        # time_series = []
        pass

        # TODO: Test cases
        # test a few different time series for this case
        # test a failed case where window isn't found (raises exception)
