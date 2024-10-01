import unittest

import numpy as np

from seer.anomaly_detection.models.timeseries_anomalies import MPTimeSeriesAnomalies


class TestConverters(unittest.TestCase):

    def setUp(self):
        self.anomalies = MPTimeSeriesAnomalies(
            flags=["none", "none"],
            scores=[0.8, 0.9],
            matrix_profile=np.array([[0.1, 0, 1, 2], [0.2, 1, 2, 3]]),
            window_size=2,
            thresholds=[0.0, 0.0],
        )

    def test_get_anomaly_algo_data_with_padding(self):
        # Test with front_pad_to_len greater than the length of matrix_profile
        front_pad_to_len = 5
        expected_data = [
            None,
            None,
            None,
            {"dist": 0.1, "idx": 0, "l_idx": 1, "r_idx": 2},
            {"dist": 0.2, "idx": 1, "l_idx": 2, "r_idx": 3},
        ]

        algo_data = self.anomalies.get_anomaly_algo_data(front_pad_to_len)

        assert algo_data == expected_data

    def test_get_anomaly_algo_data_without_padding(self):
        # Test with front_pad_to_len equal to the length of matrix_profile
        front_pad_to_len = 2
        expected_data = [
            {"dist": 0.1, "idx": 0, "l_idx": 1, "r_idx": 2},
            {"dist": 0.2, "idx": 1, "l_idx": 2, "r_idx": 3},
        ]

        algo_data = self.anomalies.get_anomaly_algo_data(front_pad_to_len)
        assert algo_data == expected_data

        front_pad_to_len = 1
        algo_data = self.anomalies.get_anomaly_algo_data(front_pad_to_len)
        assert algo_data == expected_data
