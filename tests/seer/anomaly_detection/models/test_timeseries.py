import unittest

import numpy as np

from seer.anomaly_detection.models.timeseries import MPTimeSeries
from seer.anomaly_detection.models.timeseries_anomalies import MPTimeSeriesAnomalies


class TestTimeSeries(unittest.TestCase):

    def setUp(self):
        self.ts_with_anomalies = MPTimeSeries(
            timestamps=np.array([1.0, 2.0, 3.0]),
            values=np.array([10.0, 20.0, 30.0]),
            anomalies=MPTimeSeriesAnomalies(
                flags=["none"],
                scores=[0.8],
                matrix_profile=np.array([[0.1, 0, 1, 2]]),
                window_size=2,
            ),
        )

        self.ts_without_anomalies = MPTimeSeries(
            timestamps=np.array([1.0, 2.0, 3.0]),
            values=np.array([10.0, 20.0, 30.0]),
            anomalies=None,
        )

    def test_get_anomaly_algo_data_with_anomalies(self):
        expected_result = {"window_size": 2}
        result = self.ts_with_anomalies.get_anomaly_algo_data()
        assert result == expected_result

    def test_get_anomaly_algo_data_without_anomalies(self):
        expected_result = None
        result = self.ts_without_anomalies.get_anomaly_algo_data()
        assert result == expected_result
