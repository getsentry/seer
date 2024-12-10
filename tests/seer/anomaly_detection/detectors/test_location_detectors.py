import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from seer.anomaly_detection.detectors.location_detectors import (
    LinearRegressionLocationDetector,
    PointLocation,
    ProphetLocationDetector,
)
from seer.anomaly_detection.models import AlgoConfig, ThresholdType


class TestLinearRegressionLocationDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LinearRegressionLocationDetector(window_size=5, threshold=0.5)

    def test_increasing_trend(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 6.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps).location,
            PointLocation.UP,
        )

    def test_decreasing_trend(self):
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 0.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps).location,
            PointLocation.DOWN,
        )

    def test_stable_trend(self):
        values = np.array([2.0, 2.1, 1.9, 2.2, 2.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 2.1
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps).location,
            PointLocation.NONE,
        )

    def test_insufficient_data(self):
        values = np.array([1.0, 2.0])
        timestamps = np.array([1.0, 2.0])
        cur_val = 3.0
        cur_timestamp = 3.0
        self.assertEqual(self.detector.detect(cur_val, cur_timestamp, values, timestamps), None)

    def test_custom_window_size(self):
        detector = LinearRegressionLocationDetector(window_size=3, threshold=0.5)
        values = np.array([1.0, 2.0, 3.0])
        timestamps = np.array([1.0, 2.0, 3.0])
        cur_val = 4.0
        cur_timestamp = 4.0
        self.assertEqual(
            detector.detect(cur_val, cur_timestamp, values, timestamps).location, PointLocation.UP
        )

    def test_edge_case_all_same_values(self):
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 1.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps).location,
            PointLocation.NONE,
        )

    def test_edge_case_single_spike(self):
        values = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur_val = 1.0
        cur_timestamp = 6.0
        self.assertEqual(
            self.detector.detect(cur_val, cur_timestamp, values, timestamps).location,
            PointLocation.NONE,
        )


class TestProphetLocationDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ProphetLocationDetector()
        self.algo_config = AlgoConfig(
            mp_ignore_trivial=True,
            mp_normalize=False,
            prophet_uncertainty_samples=25,
            prophet_mcmc_samples=0,
            mp_fixed_window_size=10,
            return_thresholds=False,
            return_predicted_range=False,
        )

    @patch("prophet.Prophet.predict")
    @patch("prophet.Prophet.fit")
    def test_detect_up(self, mock_fit, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "yhat_upper": [1.5, 2.5, 3.5],
                "yhat_lower": [0.5, 1.5, 2.5],
            }
        )

        result = self.detector.detect(
            streamed_value=4.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
            algo_config=self.algo_config,
        )

        mock_predict.assert_called_once()
        mock_fit.assert_called_once()
        self.assertEqual(result.location, PointLocation.UP)
        self.assertEqual(result.thresholds, [])

    @patch("prophet.Prophet.predict")
    @patch("prophet.Prophet.fit")
    def test_detect_down(self, mock_fit, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "yhat_upper": [1.5, 2.5, 3.5],
                "yhat_lower": [0.5, 1.5, 2.5],
            }
        )
        result = self.detector.detect(
            streamed_value=0.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
            algo_config=self.algo_config,
        )

        mock_predict.assert_called_once()
        mock_fit.assert_called_once()
        self.assertEqual(result.location, PointLocation.DOWN)
        self.assertEqual(result.thresholds, [])

    @patch("prophet.Prophet.predict")
    @patch("prophet.Prophet.fit")
    def test_detect_none(self, mock_fit, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "yhat_upper": [1.5, 2.5, 3.5],
                "yhat_lower": [0.5, 1.5, 2.5],
            }
        )
        self.algo_config.return_thresholds = False
        self.algo_config.return_predicted_range = False

        result = self.detector.detect(
            streamed_value=3.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
            algo_config=self.algo_config,
        )

        mock_predict.assert_called_once()
        mock_fit.assert_called_once()
        self.assertEqual(result.location, PointLocation.NONE)
        self.assertEqual(result.thresholds, [])

    @patch("prophet.Prophet.predict")
    @patch("prophet.Prophet.fit")
    def test_detect_up_without_uncertainty_samples(self, mock_fit, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "yhat_upper": [1.5, 2.5, 3.5],
                "yhat_lower": [0.5, 1.5, 2.5],
            }
        )
        self.algo_config.return_thresholds = False
        self.algo_config.prophet_uncertainty_samples = 0
        self.algo_config.prophet_mcmc_samples = 0
        self.algo_config.return_predicted_range = False

        result = self.detector.detect(
            streamed_value=4.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
            algo_config=self.algo_config,
        )

        mock_predict.assert_called_once()
        mock_fit.assert_called_once()
        self.assertEqual(result.location, PointLocation.UP)
        self.assertEqual(result.thresholds, [])

    @patch("prophet.Prophet.predict")
    @patch("prophet.Prophet.fit")
    def test_detect_down_without_uncertainty_samples(self, mock_fit, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
            }
        )

        self.algo_config.prophet_uncertainty_samples = 0
        self.algo_config.prophet_mcmc_samples = 0
        detector = ProphetLocationDetector()
        result = detector.detect(
            streamed_value=0.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
            algo_config=self.algo_config,
        )
        mock_predict.assert_called_once()
        mock_fit.assert_called_once()
        self.assertEqual(result.location, PointLocation.DOWN)
        self.assertEqual(result.thresholds, [])

    @patch("prophet.Prophet.predict")
    @patch("prophet.Prophet.fit")
    def test_detect_none_without_uncertainty_samples(self, mock_fit, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "yhat_upper": [1.5, 2.5, 3.5],
                "yhat_lower": [0.5, 1.5, 2.5],
            }
        )

        self.algo_config.prophet_uncertainty_samples = 0
        self.algo_config.prophet_mcmc_samples = 0
        result = self.detector.detect(
            streamed_value=3.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
            algo_config=self.algo_config,
        )

        mock_predict.assert_called_once()
        mock_fit.assert_called_once()
        self.assertEqual(result.location, PointLocation.NONE)

    @patch("prophet.Prophet.predict")
    @patch("prophet.Prophet.fit")
    def test_detect_down_with_thresholds(self, mock_fit, mock_predict):
        y_hat_upper = 3.5
        y_hat_lower = 2.5
        trend_upper = 3.6
        trend_lower = 2.6
        mock_predict.return_value = pd.DataFrame(
            {
                "ds": [1.0, 2.0, 3.0],
                "yhat": [1.0, 2.0, 3.0],
                "yhat_upper": [1.5, 2.5, y_hat_upper],
                "yhat_lower": [0.5, 1.5, y_hat_lower],
                "trend_upper": [1.5, 2.5, trend_upper],
                "trend_lower": [0.5, 1.5, trend_lower],
            }
        )

        self.algo_config.return_thresholds = True
        self.algo_config.return_predicted_range = True
        detector = ProphetLocationDetector()
        result = detector.detect(
            streamed_value=0.0,
            streamed_timestamp=3.0,
            history_values=np.array([1.0, 2.0]),
            history_timestamps=np.array([1.0, 2.0]),
            algo_config=self.algo_config,
        )
        mock_predict.assert_called_once()
        mock_fit.assert_called_once()
        self.assertEqual(result.location, PointLocation.DOWN)
        self.assertEqual(len(result.thresholds), 2)
        self.assertEqual(result.thresholds[0].type, ThresholdType.PREDICTION)
        self.assertEqual(result.thresholds[0].upper, y_hat_upper)
        self.assertEqual(result.thresholds[0].lower, y_hat_lower)
        self.assertEqual(result.thresholds[1].type, ThresholdType.TREND)
        self.assertEqual(result.thresholds[1].upper, trend_upper)
        self.assertEqual(result.thresholds[1].lower, trend_lower)
