import unittest

from seer.anomaly_detection.detectors.anomaly_detectors import (
    MPBatchAnomalyDetector,
    MPStreamAnomalyDetector,
)


class TestMPBatchAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MPBatchAnomalyDetector()

    def test_detect(self):
        # TODO: Import time series
        # time_series = []
        pass

        # TODO: Test cases


class TestMPStreamAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MPStreamAnomalyDetector()

    def test_optimal_window_size(self):
        # TODO: Import time series
        # time_series = []
        pass

        # TODO: Test cases
