import unittest
import pandas as pd
import pickle

from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from seer.anomaly_detection.prophet_detector import ProphetDetector, ProphetParams


class TestProphetDetector(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = "tests/data"
        self.sentry_data_df = pd.DataFrame([
            {'time': 1644350400, 'count': 1},
            {'time': 1644352400, 'count': 1},
            {'time': 1644358400, 'count': 2},
            {'time': 1644361400, 'count': 1},
            {'time': 1644362400, 'count': 2},
            {'time': 1644366400, 'count': 1},
        ])
        self.start = "2022-02-06 19:59:40"
        self.end = "2022-02-15 20:04:39"
        self.granularity = 300

        prophet_params = ProphetParams()
        self.prophet_detector = ProphetDetector(prophet_params)

    def save_object(self, obj, filename):
        with open(f"{self.test_data_dir}/{filename}", "wb") as file:  # Overwrites any existing file.
            pickle.dump(obj, file, -1)

    @staticmethod
    def read_pickle_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_pre_process_data(self):
        self.prophet_detector.pre_process_data(self.sentry_data_df, self.granularity, self.start, self.end)
        expected_output = pd.read_pickle(f"{self.test_data_dir}/pre_processed_df.pkl")

        assert_frame_equal(expected_output, self.prophet_detector.train)

    def test_fit(self):
        self.prophet_detector.train = pd.read_pickle(f"{self.test_data_dir}/pre_processed_df.pkl")
        self.prophet_detector.fit()
        actual_params = self.prophet_detector.model.params

        f = open(f"{self.test_data_dir}/prophet_detector_model.pkl", "rb")
        expected_model = pickle.load(f)
        expected_params = expected_model.params

        # cannot do a dict comparison since values are numpy arrays
        for k, v in actual_params.items():
            assert_array_almost_equal(v, expected_params.get(k), decimal=3)

        assert expected_model.start == self.prophet_detector.model.start

    def test_predict(self):
        self.prophet_detector.test = self.read_pickle_file(f"{self.test_data_dir}/test_df.pkl")
        self.prophet_detector.model = self.read_pickle_file(f"{self.test_data_dir}/prophet_detector_model.pkl")
        expected_df_with_forecasts = self.read_pickle_file(f"{self.test_data_dir}/expected_df_with_forecasts.pkl")

        df_with_forecasts = self.prophet_detector.predict()
        assert_frame_equal(df_with_forecasts, expected_df_with_forecasts, exact_check=False, check_less_precise=True)

    def test_add_prophet_uncertainty(self):
        self.prophet_detector.model = self.read_pickle_file(f"{self.test_data_dir}/prophet_detector_model.pkl")
        self.prophet_detector.bc_lambda = -5.466435161879652
        df_with_forecasts = self.read_pickle_file(f"{self.test_data_dir}/expected_df_with_forecasts.pkl")
        expected_df_with_uncertainty = self.read_pickle_file(f"{self.test_data_dir}/df_with_uncertainty.pkl")

        actual_df_with_uncertainty = self.prophet_detector.add_prophet_uncertainty(df_with_forecasts)

        assert_frame_equal(expected_df_with_uncertainty, actual_df_with_uncertainty, exact_check=False, check_less_precise=True)

    def test_add_scale_score(self):
        df_with_uncertainty = self.read_pickle_file(f"{self.test_data_dir}/df_with_uncertainty.pkl")
        self.prophet_detector.start, self.prophet_detector.end = self.start, self.end
        expected_df_with_anomalies = self.read_pickle_file(f"{self.test_data_dir}/df_with_anomalies.pkl")

        actual_df_with_anomalies = self.prophet_detector.scale_scores(df_with_uncertainty)

        assert_frame_equal(actual_df_with_anomalies, expected_df_with_anomalies, exact_check=False, check_less_precise=True)

    def test_boxcox(self):
        y_before_boxcox = self.read_pickle_file(f"{self.test_data_dir}/train_y_before_boxcox.pkl")
        expected_y_after_boxcox = self.read_pickle_file(f"{self.test_data_dir}/train_y_after_boxcox.pkl")

        actual_y_after_boxcox = self.prophet_detector._boxcox(y_before_boxcox)

        assert_series_equal(actual_y_after_boxcox, expected_y_after_boxcox)

    def test_inv_boxcox(self):
        y_after_boxcox = self.read_pickle_file(f"{self.test_data_dir}/train_y_after_boxcox.pkl")
        self.prophet_detector.bc_lambda = -5.466435161879652
        expected_y_after_inv_bocox = self.read_pickle_file(f"{self.test_data_dir}/y_after_inv_boxcox.pkl")

        actual_y_after_inv_bocox = self.prophet_detector._inv_boxcox(y_after_boxcox)

        assert_series_equal(actual_y_after_inv_bocox, expected_y_after_inv_bocox)
