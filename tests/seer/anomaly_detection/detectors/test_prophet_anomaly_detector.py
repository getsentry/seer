import unittest

import numpy as np
import pandas as pd

from seer.anomaly_detection.detectors.prophet_anomaly_detector import ProphetAnomalyDetector
from seer.anomaly_detection.models import AlgoConfig


class TestProphetAnomalyDetector(unittest.TestCase):
    def setUp(self):
        """Set up test data and detector instance"""
        self.detector = ProphetAnomalyDetector()

        # Create sample timeseries data
        self.dates = pd.date_range(start="2023-01-01", periods=100, freq="15min", tz="UTC")
        # self.timestamps = np.array(self.dates.astype(np.int64) // 10**9, dtype=np.float64)

        self.timestamps = np.array([date.timestamp() for date in self.dates])

        # Create sinusoidal pattern with daily seasonality
        t = np.linspace(0, 4 * np.pi, 100)
        self.values = np.sin(t) * 10 + 50

        # Add some noise
        np.random.seed(42)
        self.values = self.values + np.random.normal(0, 1, 100)
        self.values = np.array(self.values, dtype=np.float64)

        self.algo_config = AlgoConfig(
            mp_ignore_trivial=False,
            mp_normalize=False,
            mp_fixed_window_size=10,
            prophet_uncertainty_samples=25,
            prophet_mcmc_samples=0,
            return_thresholds=False,
            return_predicted_range=False,
        )

    def test_predict_basic_functionality(self):
        """Test basic prediction functionality"""
        forecast = self.detector.predict(
            timestamps=self.timestamps,
            values=self.values,
            forecast_len=10,
            time_period=15,
            sensitivity="medium",
            algo_config=self.algo_config,
        )

        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertTrue("yhat" in forecast.columns)
        self.assertTrue("yhat_lower" in forecast.columns)
        self.assertTrue("yhat_upper" in forecast.columns)
        self.assertTrue("actual" in forecast.columns)
        self.assertEqual(len(forecast), len(self.timestamps) + 10)

    def test_predict_with_invalid_data(self):
        """Test prediction with invalid data"""
        # Test with infinite values
        values_with_inf = self.values.copy()
        values_with_inf[0] = np.inf

        with self.assertRaises(ValueError):
            self.detector.predict(
                timestamps=self.timestamps,
                values=values_with_inf,
                forecast_len=10,
                time_period=15,
                sensitivity="medium",
                algo_config=self.algo_config,
            )

    def test_boxcox_transformation(self):
        """Test Box-Cox transformation methods"""
        test_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test forward transformation
        transformed, lambda_param = self.detector._boxcox(test_values)
        self.assertEqual(len(transformed), len(test_values))

        # Test inverse transformation
        inverse_transformed = self.detector._inv_boxcox(transformed, lambda_param)
        np.testing.assert_array_almost_equal(test_values, inverse_transformed, decimal=4)

    def test_different_sensitivities(self):
        """Test predictions with different sensitivity levels"""

        forecasts = {}
        for sensitivity in ["low", "medium", "high"]:
            forecast = self.detector.predict(
                timestamps=self.timestamps,
                values=self.values,
                forecast_len=10,
                time_period=15,
                sensitivity=sensitivity,
                algo_config=self.algo_config,
            )
            forecasts[sensitivity] = forecast

        # Higher sensitivity should have narrower prediction intervals
        high_interval = forecasts["high"]["yhat_upper"] - forecasts["high"]["yhat_lower"]
        low_interval = forecasts["low"]["yhat_upper"] - forecasts["low"]["yhat_lower"]

        self.assertTrue((high_interval <= low_interval).all())

    def test_different_time_periods(self):
        """Test predictions with different time periods"""
        for period in [15, 30, 60]:
            forecast = self.detector.predict(
                timestamps=self.timestamps,
                values=self.values,
                forecast_len=10,
                time_period=period,
                sensitivity="medium",
                algo_config=self.algo_config,
            )

            self.assertIsInstance(forecast, pd.DataFrame)
            self.assertTrue(len(forecast) >= 10)

    def test_pre_process_data(self):
        """Test data preprocessing"""
        df = pd.DataFrame({"ds": self.dates, "y": self.values})

        processed_df, bc_lambda = self.detector._pre_process_data(df, granularity=15)

        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertIsInstance(bc_lambda, float)
        self.assertEqual(len(processed_df), len(df))
        self.assertTrue(all(~np.isnan(processed_df["y"])))
        self.assertTrue(all(~np.isinf(processed_df["y"])))
