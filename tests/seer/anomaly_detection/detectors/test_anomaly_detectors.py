import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from seer.anomaly_detection.detectors.anomaly_detectors import (
    MPBatchAnomalyDetector,
    MPStreamAnomalyDetector,
)
from seer.anomaly_detection.detectors.mp_config import MPConfig
from seer.anomaly_detection.models import MPTimeSeriesAnomalies
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.anomaly_detection.models.timeseries import TimeSeries
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestMPBatchAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MPBatchAnomalyDetector()

        self.config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="up", expected_seasonality="auto"
        )  # TODO: Placeholder values as not used in detection yet

        self.mp_config = MPConfig(ignore_trivial=False, normalize_mp=False)
        self.ws_selector = MagicMock()
        self.scorer = MagicMock()
        self.mp_utils = MagicMock()

    @patch("stumpy.stump")
    def test_compute_matrix_profile(self, mock_stump):

        # Mock to return dummy values
        mock_stump.return_value = np.array([1, 2, 3, 4])
        self.scorer.batch_score = MagicMock(return_value=([], []))

        timeseries, mp_dists, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        for ts_values, mp_dist_baseline, window_size in zip(timeseries, mp_dists, window_sizes):
            ts = TimeSeries(timestamps=np.array([]), values=ts_values)

            self.ws_selector.optimal_window_size = MagicMock(return_value=window_size)

            self.mp_utils.get_mp_dist_from_mp = MagicMock(return_value=mp_dist_baseline)

            result = self.detector._compute_matrix_profile(
                ts,
                self.config,
                ws_selector=self.ws_selector,
                mp_config=self.mp_config,
                scorer=self.scorer,
                mp_utils=self.mp_utils,
            )

            self.assertIsInstance(result, MPTimeSeriesAnomalies)
            self.assertIsInstance(result.flags, list)
            self.assertIsInstance(result.scores, list)
            self.assertIsInstance(result.matrix_profile, np.ndarray)
            self.assertIsInstance(result.window_size, int)


class TestMPStreamAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MPStreamAnomalyDetector(
            base_timestamps=np.array([1, 2, 3]),
            base_values=np.array([1.0, 2.0, 3.0]),
            base_mp=np.array([[0.1], [0.2], [0.3]]),
            window_size=2,
        )
        self.timeseries = TimeSeries(
            timestamps=np.array([1, 2, 3]), values=np.array([1.1, 2.1, 3.1])
        )
        self.config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="up", expected_seasonality="auto"
        )  # TODO: Placeholder values as not used in detection yet

    @patch("stumpy.stumpi")
    @patch("seer.anomaly_detection.detectors.MPScorer")
    @patch("seer.anomaly_detection.detectors.MPUtils")
    def test_detect(self, MockMPUtils, MockMPScorer, MockStumpi):
        mock_stream = MagicMock()
        MockStumpi.return_value = mock_stream
        mock_scorer = MockMPScorer.return_value
        mock_utils = MockMPUtils.return_value

        mock_stream.P_ = [[0.1], [0.2], [0.3]]
        mock_stream.I_ = [[0], [1], [2]]
        mock_stream.left_I_ = [[0], [1], [2]]
        mock_stream.T_ = np.array([1.1, 2.1, 3.1])

        mock_utils.get_mp_dist_from_mp.return_value = np.array([0.1, 0.2])

        mock_scorer.stream_score.return_value = ([0.5], ["none"])

        anomalies = self.detector.detect(self.timeseries, self.config, mock_scorer, mock_utils)

        self.assertIsInstance(anomalies, MPTimeSeriesAnomalies)
        self.assertIsInstance(anomalies.flags, list)
        self.assertIsInstance(anomalies.scores, list)
        self.assertIsInstance(anomalies.matrix_profile, np.ndarray)
        self.assertIsInstance(anomalies.window_size, int)

        self.assertEqual(len(anomalies.scores), 3)
        self.assertEqual(len(anomalies.flags), 3)
        self.assertEqual(len(anomalies.matrix_profile), 3)
        mock_scorer.stream_score.assert_called()
