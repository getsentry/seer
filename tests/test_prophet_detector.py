import unittest
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from seer.anomaly_detection.prophet_detector import ProphetDetector
from seer.anomaly_detection.prophet_params import ProphetParams


class TestProphetDetector(unittest.TestCase):
    def setUp(self):
        self.start = "2022-02-06 19:59:40"
        self.end = "2022-02-15 20:04:39"
        self.granularity = 300

        self.test_data_dir = "tests/data"

        prophet_params = ProphetParams()
        self.prophet_detector = ProphetDetector(prophet_params)

    def save_object(self, obj, filename):
        with open(
            f"{self.test_data_dir}/{filename}", "wb"
        ) as file:  # Overwrites any existing file.
            pickle.dump(obj, file, -1)

    @staticmethod
    def read_pickle_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_pre_process_data(self):
        input_data = pd.DataFrame(
            [
                {"time": 1644350400, "count": 1},
                {"time": 1644350700, "count": 1},
                {"time": 1644351000, "count": 2},
                {"time": 1644351300, "count": 1},
                {"time": 1644351600, "count": 2},
                {"time": 1644351900, "count": 1},
            ]
        )
        expected = pd.DataFrame(
            [
                {"ds": pd.Timestamp("2022-02-08 20:00:00"), "y": 0.7862882049762442},
                {"ds": pd.Timestamp("2022-02-08 20:05:00"), "y": 0.7895432699110309},
                {"ds": pd.Timestamp("2022-02-08 20:10:00"), "y": 0.8131050821884428},
                {"ds": pd.Timestamp("2022-02-08 20:15:00"), "y": 0.85109721550428},
                {"ds": pd.Timestamp("2022-02-08 20:20:00"), "y": 0.8907101491544074},
                {"ds": pd.Timestamp("2022-02-08 20:25:00"), "y": 0.9179890446650987},
            ]
        )
        self.prophet_detector.pre_process_data(
            input_data, self.granularity, self.start, self.end
        )
        actual = self.prophet_detector.train.reset_index(drop=True)
        assert_frame_equal(expected, actual)


    def test_pre_process_data_constant(self):
        input_data = pd.DataFrame(
            [
                {"time": 1644350400, "count": 1.0},
                {"time": 1644350700, "count": 1.0},
                {"time": 1644351000, "count": 1.0},
                {"time": 1644351300, "count": 1.0},
                {"time": 1644351600, "count": 1.0},
                {"time": 1644351900, "count": 1.0},
            ]
        )
        expected = pd.DataFrame(
            [
                {"ds": pd.Timestamp("2022-02-08 20:00:00"), "y": 1.0},
                {"ds": pd.Timestamp("2022-02-08 20:05:00"), "y": 1.0},
                {"ds": pd.Timestamp("2022-02-08 20:10:00"), "y": 1.0},
                {"ds": pd.Timestamp("2022-02-08 20:15:00"), "y": 1.0},
                {"ds": pd.Timestamp("2022-02-08 20:20:00"), "y": 1.0},
                {"ds": pd.Timestamp("2022-02-08 20:25:00"), "y": 1.0},
            ]
        )
        self.prophet_detector.pre_process_data(
            input_data, self.granularity, self.start, self.end
        )
        actual = self.prophet_detector.train.reset_index(drop=True)
        assert_frame_equal(expected, actual)

    def test_pre_process_data_gaps(self):
        input_data = pd.DataFrame(
            [
                {"time": 1644350400, "count": 1},
                {"time": 1644350700, "count": 1},
                {"time": 1644351000, "count": 2},  # missing a record here
                {"time": 1644351600, "count": 2},
                {"time": 1644351900, "count": 1},
            ]
        )
        expected = pd.DataFrame(
            [
                {"ds": pd.Timestamp("2022-02-08 20:00:00"), "y": 0.9457422784316951},
                {"ds": pd.Timestamp("2022-02-08 20:05:00"), "y": 1.0138675317692367},
                {"ds": pd.Timestamp("2022-02-08 20:10:00"), "y": 1.136321690202418},
                {"ds": pd.Timestamp("2022-02-08 20:15:00"), "y": np.nan},
                {"ds": pd.Timestamp("2022-02-08 20:20:00"), "y": 1.255002928134423},
                {"ds": pd.Timestamp("2022-02-08 20:25:00"), "y": 1.3179081436029578},
            ]
        )

        self.prophet_detector.pre_process_data(
            input_data, self.granularity, self.start, self.end
        )
        actual = self.prophet_detector.train.reset_index(drop=True)
        assert_frame_equal(expected, actual)

    def test_fit(self):
        expected_params = {
            "k": np.array([[0.34004261]]),
            "m": np.array([[0.71435621]]),
            "delta": np.array([[1.97169826e-06, -8.64940102e-02, -6.63167128e-09]]),
            "sigma_obs": np.array([[0.00809842]]),
            "beta": np.array([[-5.86725629e-17]]),
            "trend": np.array([[0.71435621, 0.78236473, 0.85037365, 0.95179388, 1.00250399]]),
        }

        input_data = pd.DataFrame(
            [
                {"ds": pd.Timestamp("2022-02-08 20:00:00"), "y": 0.9457422784316951},
                {"ds": pd.Timestamp("2022-02-08 20:05:00"), "y": 1.0138675317692367},
                {"ds": pd.Timestamp("2022-02-08 20:10:00"), "y": 1.136321690202418},
                {"ds": pd.Timestamp("2022-02-08 20:15:00"), "y": np.nan},
                {"ds": pd.Timestamp("2022-02-08 20:20:00"), "y": 1.255002928134423},
                {"ds": pd.Timestamp("2022-02-08 20:25:00"), "y": 1.3179081436029578},
            ]
        )
        self.prophet_detector.train = input_data
        self.prophet_detector.fit()
        actual_params = self.prophet_detector.model.params

        # cannot do a dict comparison since values are numpy arrays
        for k, v in actual_params.items():
            assert_array_almost_equal(v, expected_params.get(k), decimal=1)

    def test_predict(self):
        self.prophet_detector.test = pd.DataFrame(
            [
                {"ds": pd.Timestamp("2022-02-08 20:00:00"), "y": 0.9457422784316951},
                {"ds": pd.Timestamp("2022-02-08 20:05:00"), "y": 1.0138675317692367},
                {"ds": pd.Timestamp("2022-02-08 20:10:00"), "y": 1.136321690202418},
                {"ds": pd.Timestamp("2022-02-08 20:15:00"), "y": np.nan},
                {"ds": pd.Timestamp("2022-02-08 20:20:00"), "y": 1.255002928134423},
                {"ds": pd.Timestamp("2022-02-08 20:25:00"), "y": 1.3179081436029578},
            ]
        )
        expected_output = pd.DataFrame(
            [
                {"ds": pd.Timestamp("2022-02-08 20:00:00"), "yhat": 0.9414558634589312},
                {"ds": pd.Timestamp("2022-02-08 20:05:00"), "yhat": 1.0310848494772002},
                {"ds": pd.Timestamp("2022-02-08 20:10:00"), "yhat": 1.1207143551989083},
                {"ds": pd.Timestamp("2022-02-08 20:15:00"), "yhat": 1.187545628844418},
                {"ds": pd.Timestamp("2022-02-08 20:20:00"), "yhat": 1.2543769024899278},
                {"ds": pd.Timestamp("2022-02-08 20:25:00"), "yhat": 1.3212081743874506},
            ]
        )

        self.prophet_detector.model = self.read_pickle_file(
            f"{self.test_data_dir}/prophet_detector_model.pkl"
        )

        actual_output = self.prophet_detector.predict()[["ds", "yhat"]].reset_index(drop=True)
        assert_frame_equal(expected_output, actual_output, check_exact=False)

    def test_add_prophet_uncertainty(self):
        self.prophet_detector.model = self.read_pickle_file(
            f"{self.test_data_dir}/prophet_detector_model.pkl"
        )
        self.prophet_detector.bc_lambda = 0.5719605326696966
        input_data = pd.DataFrame(
            [
                {
                    "ds": pd.Timestamp("2022-02-08 20:00:00"),
                    "yhat": 1.1237532636927319,
                    "y": 1.1296738745089212,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:05:00"),
                    "yhat": 1.2490179414050782,
                    "y": 1.2247169487634437,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:10:00"),
                    "yhat": 1.3773434566712188,
                    "y": 1.3999999999999995,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:15:00"),
                    "yhat": 1.4750050070362501,
                    "y": 0.0,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:20:00"),
                    "yhat": 1.5743446089907094,
                    "y": 1.575283051236556,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:25:00"),
                    "yhat": 1.6753527818165614,
                    "y": 1.670326125491079,
                },
            ]
        )

        expected_output = pd.DataFrame(
            [
                {
                    "ds": pd.Timestamp("2022-02-08 20:00:00"),
                    "yhat": 1.3817477059151049,
                    "yhat_upper": 1.4009384131879163,
                    "yhat_lower": 1.3611499336630515,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:05:00"),
                    "yhat": 1.5663172629067095,
                    "yhat_upper": 1.587198443034377,
                    "yhat_lower": 1.5453607624439103,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:10:00"),
                    "yhat": 1.7614777573253377,
                    "yhat_upper": 1.7827218351868614,
                    "yhat_lower": 1.7397837277372874,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:15:00"),
                    "yhat": 1.9140883176540768,
                    "yhat_upper": 1.9366295374976268,
                    "yhat_lower": 1.892963995341399,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:20:00"),
                    "yhat": 2.0729118290491595,
                    "yhat_upper": 2.0955803315944896,
                    "yhat_lower": 2.0503286313988625,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:25:00"),
                    "yhat": 2.2380878769994452,
                    "yhat_upper": 2.260502430066593,
                    "yhat_lower": 2.2156922042426763,
                },
            ]
        )

        actual_output = self.prophet_detector.add_prophet_uncertainty(input_data)[
            ["ds", "yhat", "yhat_upper", "yhat_lower"]
        ].reset_index(drop=True)

        assert_frame_equal(expected_output, actual_output, check_exact=False, rtol=1e-2)

    def test_add_scale_score(self):
        input_data = pd.DataFrame(
            [
                {
                    "ds": pd.Timestamp("2022-02-08 20:00:00"),
                    "yhat": 1.3817477059151049,
                    "yhat_upper": 1.4023691957335735,
                    "yhat_lower": 1.3615256964012832,
                    "y": 1.390338386632183,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:05:00"),
                    "yhat": 1.5663172629067095,
                    "yhat_upper": 1.5873156956246217,
                    "yhat_lower": 1.546587809817026,
                    "y": 1.53005085131137,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:10:00"),
                    "yhat": 1.7614777573253377,
                    "yhat_upper": 1.7819664967553055,
                    "yhat_lower": 1.7394097921548797,
                    "y": 1.796568718078606,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:15:00"),
                    "yhat": 1.9140883176540768,
                    "yhat_upper": 1.93427473503105,
                    "yhat_lower": 1.891795731356615,
                    "y": 0.0,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:20:00"),
                    "yhat": 2.0729118290491595,
                    "yhat_upper": 2.093986589832242,
                    "yhat_lower": 2.0506369565555422,
                    "y": 2.0744293829613683,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:25:00"),
                    "yhat": 2.2380878769994452,
                    "yhat_upper": 2.261380596664887,
                    "yhat_lower": 2.215264179388562,
                    "y": 2.2297804949840363,
                },
            ]
        )
        input_data.index = input_data["ds"]

        expected_output = pd.DataFrame(
            [
                {
                    "ds": pd.Timestamp("2022-02-08 20:00:00"),
                    "final_score": 0.471853070257689,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:05:00"),
                    "final_score": 0.3901929508487105,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:10:00"),
                    "final_score": 0.3901929508487105,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:15:00"),
                    "final_score": 0.3901929508487105,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:20:00"),
                    "final_score": 0.3901929508487105,
                },
                {
                    "ds": pd.Timestamp("2022-02-08 20:25:00"),
                    "final_score": 0.3901929508487105,
                },
            ]
        )

        self.prophet_detector.start = datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
        self.prophet_detector.end = datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")

        actual_output = self.prophet_detector.scale_scores(input_data)[
            ["ds", "final_score"]
        ].reset_index(drop=True)

        assert_frame_equal(expected_output, actual_output, check_exact=False)

    def test_boxcox(self):
        input_data = [
            1.390338386632183,
            1.53005085131137,
            1.796568718078606,
            0.0,
            2.0744293829613683,
            2.2297804949840363,
        ]
        expected_output = [3.2978869, 3.88122086, 5.14366647, 0.0, 6.67962632, 7.64087904]

        actual_output = self.prophet_detector._boxcox(pd.Series(input_data))

        assert_array_almost_equal(expected_output, actual_output, decimal=3)

    def test_inv_boxcox(self):
        input_data = [3.2978869, 3.88122086, 5.14366647, 0.0, 6.67962632, 7.64087904]
        expected_output = [
            1.390338386632183,
            1.53005085131137,
            1.796568718078606,
            0.0,
            2.0744293829613683,
            2.2297804949840363,
        ]

        self.prophet_detector.bc_lambda = 2.5874070835521463
        actual_output = self.prophet_detector._inv_boxcox(pd.Series(input_data))

        assert_array_almost_equal(expected_output, list(actual_output), decimal=3)
