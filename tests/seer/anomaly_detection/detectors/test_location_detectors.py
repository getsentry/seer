import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from seer.anomaly_detection.detectors.location_detectors import (
    LinearRegressionLocationDetector,
    PointLocation,
    ProphetLocationDetector,
)


class TestLinearRegressionLocationDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LinearRegressionLocationDetector(window_size=5, threshold=0.5)

    def test_increasing_trend(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 6.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps), PointLocation.UP
        )

    def test_decreasing_trend(self):
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 0.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps),
            PointLocation.DOWN,
        )

    def test_stable_trend(self):
        values = np.array([2.0, 2.1, 1.9, 2.2, 2.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 2.1
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps), PointLocation.NONE
        )

    def test_insufficient_data(self):
        values = np.array([1.0, 2.0])
        timestamps = np.array([1.0, 2.0])
        cur_val = 3.0
        cur_timestamp = 3.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps), PointLocation.NONE
        )

    def test_custom_window_size(self):
        detector = LinearRegressionLocationDetector(window_size=3, threshold=0.5)
        values = np.array([1.0, 2.0, 3.0])
        timestamps = np.array([1.0, 2.0, 3.0])
        cur_val = 4.0
        cur_timestamp = 4.0
        self.assertEqual(
            detector.detect(cur_val, cur_timestamp, values, timestamps), PointLocation.UP
        )

    def test_edge_case_all_same_values(self):
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 1.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps), PointLocation.NONE
        )

    def test_edge_case_single_spike(self):
        values = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 1.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps), PointLocation.NONE
        )


class TestProphetLocationDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ProphetLocationDetector()

    @patch("prophet.Prophet.predict")
    def test_detect_up(self, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "trend_upper": [1.5, 2.5, 3.5],
                "trend_lower": [0.5, 1.5, 2.5],
            }
        )

        result = self.detector.detect(
            streamed_value=4.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
        )

        mock_predict.assert_called_once()
        self.assertEqual(result, PointLocation.UP)

    @patch("prophet.Prophet.predict")
    def test_detect_down(self, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "trend_upper": [1.5, 2.5, 3.5],
                "trend_lower": [0.5, 1.5, 2.5],
            }
        )

        result = self.detector.detect(
            streamed_value=0.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
        )

        mock_predict.assert_called_once()
        self.assertEqual(result, PointLocation.DOWN)

    @patch("prophet.Prophet.predict")
    def test_detect_none(self, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "trend_upper": [1.5, 2.5, 3.5],
                "trend_lower": [0.5, 1.5, 2.5],
            }
        )

        result = self.detector.detect(
            streamed_value=3.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
        )

        mock_predict.assert_called_once()
        self.assertEqual(result, PointLocation.NONE)

    @patch("prophet.Prophet.predict")
    def test_detect_up_without_uncertainty_samples(self, mock_predict):
        mock_predict.return_value = pd.DataFrame({"ds": [1.0, 2.0, 3.0], "yhat": [1.0, 2.0, 3.0]})

        result = ProphetLocationDetector(uncertainty_samples=False).detect(
            streamed_value=4.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
        )

        mock_predict.assert_called_once()
        self.assertEqual(result, PointLocation.UP)

    @patch("prophet.Prophet.predict")
    def test_detect_down_without_uncertainty_samples(self, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "trend_upper": [1.5, 2.5, 3.5],
                "trend_lower": [0.5, 1.5, 2.5],
            }
        )

        detector = ProphetLocationDetector(uncertainty_samples=False)
        result = detector.detect(
            streamed_value=0.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
        )

        mock_predict.assert_called_once()
        self.assertEqual(result, PointLocation.DOWN)

    @patch("prophet.Prophet.predict")
    def test_detect_none_without_uncertainty_samples(self, mock_prophet):
        mock_prophet.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "trend_upper": [1.5, 2.5, 3.5],
                "trend_lower": [0.5, 1.5, 2.5],
            }
        )

        result = ProphetLocationDetector(uncertainty_samples=False).detect(
            streamed_value=3.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
        )

        mock_prophet.assert_called_once()
        self.assertEqual(result, PointLocation.NONE)
