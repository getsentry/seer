import json
import os
import unittest

import numpy as np

from seer.anomaly_detection.detectors.window_size_selectors import SuSSWindowSizeSelector

# from tests.seer.anomaly_detection.timeseries.timeseries import context


class TestSuSSWindowSizeSelector(unittest.TestCase):

    def setUp(self):
        self.selector = SuSSWindowSizeSelector()

    def test_optimal_window_size(self):

        actual_windows = []

        # Check time series JSON files in test_data
        dir = "tests/seer/anomaly_detection/detectors/test_data/synthetic_series"
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)

            if os.path.isfile(f):
                if not os.path.isfile(f):
                    raise Exception("Filename is not a valid file")

                # Load json and convert to ts and mp_dist
                with open(f) as file:

                    data = json.load(file)
                    data = data["ts"]

                    ts = np.array([point["value"] for point in data], dtype=np.float64)

                    window = self.selector.optimal_window_size(ts)
                    actual_windows.append(window)

        # Check if window is within half a period
        period = 24 * 4

        for window in actual_windows:
            self.assertTrue(period / 2 <= window <= period * 1.5)
