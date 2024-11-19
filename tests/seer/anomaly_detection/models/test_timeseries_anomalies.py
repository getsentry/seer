import unittest

import numpy as np

from seer.anomaly_detection.models.timeseries_anomalies import MPTimeSeriesAnomalies


class TestConverters(unittest.TestCase):

    def setUp(self):
        self.anomalies = MPTimeSeriesAnomalies(
            flags=["none", "none"],
            scores=[0.8, 0.9],
            matrix_profile_suss=np.array([[0.1, 0, 1, 2], [0.3, 1, 2, 3]]),
            matrix_profile_fixed=np.array([[0.2, 1, 2, 3], [0.4, 2, 3, 4]]),
            window_size=2,
            thresholds=[0.0, 0.0],
            original_flags=["none", "none"],
            use_suss=[True, True],
        )

    def test_get_anomaly_algo_data_with_padding(self):
        # Test with front_pad_to_len greater than the length of matrix_profile
        front_pad_to_len = 5
        expected_data = [
            None,
            None,
            None,
            {
                "mp_suss": {"dist": 0.1, "idx": 0, "l_idx": 1, "r_idx": 2},
                "mp_fixed": {"dist": 0.2, "idx": 1, "l_idx": 2, "r_idx": 3},
                "original_flag": "none",
                "use_suss": True,
            },
            {
                "mp_suss": {"dist": 0.3, "idx": 1, "l_idx": 2, "r_idx": 3},
                "mp_fixed": {"dist": 0.4, "idx": 2, "l_idx": 3, "r_idx": 4},
                "original_flag": "none",
                "use_suss": True,
            },
        ]

        algo_data = self.anomalies.get_anomaly_algo_data(front_pad_to_len)

        assert algo_data == expected_data

    def test_get_anomaly_algo_data_without_padding(self):
        # Test with front_pad_to_len equal to the length of matrix_profile
        front_pad_to_len = 2
        expected_data = [
            {
                "mp_suss": {"dist": 0.1, "idx": 0, "l_idx": 1, "r_idx": 2},
                "mp_fixed": {"dist": 0.2, "idx": 1, "l_idx": 2, "r_idx": 3},
                "original_flag": "none",
                "use_suss": True,
            },
            {
                "mp_suss": {"dist": 0.3, "idx": 1, "l_idx": 2, "r_idx": 3},
                "mp_fixed": {"dist": 0.4, "idx": 2, "l_idx": 3, "r_idx": 4},
                "original_flag": "none",
                "use_suss": True,
            },
        ]

        algo_data = self.anomalies.get_anomaly_algo_data(front_pad_to_len)
        assert algo_data == expected_data

        front_pad_to_len = 1
        algo_data = self.anomalies.get_anomaly_algo_data(front_pad_to_len)
        assert algo_data == expected_data
