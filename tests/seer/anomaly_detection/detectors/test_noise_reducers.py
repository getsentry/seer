import unittest

import numpy as np

from seer.anomaly_detection.detectors.noise_reducers import VarianceNoiseReducer


class TestVarianceNoiseReducer(unittest.TestCase):

    def setUp(self):
        self.noise_reducer = VarianceNoiseReducer()

    def test_get_noise_parameter_standard_array(self):
        timeseries = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        window = 4
        scale_factor = 1.0
        result = self.noise_reducer.get_noise_parameter(timeseries, window, scale_factor)
        expected = 1.25
        assert result == expected

    def test_get_noise_parameter_with_scale_factor(self):
        timeseries = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        window = 4
        scale_factor = 2.0
        result = self.noise_reducer.get_noise_parameter(timeseries, window, scale_factor)
        expected = 2.5
        assert result == expected

    def test_get_noise_parameter_constant_array(self):
        timeseries = np.array([5, 5, 5, 5, 5, 5, 5, 5])
        window = 4
        result = self.noise_reducer.get_noise_parameter(timeseries, window)
        expected = 0.0
        assert result == expected

    def test_get_noise_parameter_window_larger_than_timeseries(self):
        timeseries = np.array([1, 2, 3, 4])
        window = 10
        result = self.noise_reducer.get_noise_parameter(timeseries, window)
        expected = 1.25
        assert result == expected

    def test_get_noise_parameter_empty_array(self):
        timeseries = np.array([])
        window = 4
        result = self.noise_reducer.get_noise_parameter(timeseries, window)
        expected = 0.0
        assert result == expected

    def test_get_noise_parameter_single_value(self):
        timeseries = np.array([1])
        window = 4
        result = self.noise_reducer.get_noise_parameter(timeseries, window)
        # Single value should have 0 variance
        expected = 0.0
        assert result == expected
