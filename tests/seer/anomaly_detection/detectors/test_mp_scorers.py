import unittest
from unittest.mock import patch

import numpy as np

from seer.anomaly_detection.detectors.location_detectors import PointLocation
from seer.anomaly_detection.detectors.mp_scorers import (
    LowVarianceScorer,
    MPCascadingScorer,
    MPIQRScorer,
    MPScorer,
)
from seer.anomaly_detection.models import AnomalyDetectionConfig
from seer.exceptions import ClientError
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestMPCascadingScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = MPCascadingScorer()

    def test_batch_score_synthetic_data(self):

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series",
            as_ts_datatype=False,
            include_anomaly_range=True,
        )

        expected_types = loaded_synthetic_data.expected_types
        timeseries = loaded_synthetic_data.timeseries
        timestamps = loaded_synthetic_data.timestamps
        mp_dists = loaded_synthetic_data.mp_dists
        window_sizes = loaded_synthetic_data.window_sizes
        window_starts = loaded_synthetic_data.anomaly_starts
        window_ends = loaded_synthetic_data.anomaly_ends

        threshold = 0.1

        for expected_type, ts, ts_timestamps, mp_dist, window_size, start, end in zip(
            expected_types,
            timeseries,
            timestamps,
            mp_dists,
            window_sizes,
            window_starts,
            window_ends,
        ):
            ad_config = AnomalyDetectionConfig(
                time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
            )

            flags_and_scores = self.scorer.batch_score(
                ts, ts_timestamps, mp_dist, ad_config, window_size
            )
            assert flags_and_scores is not None
            actual_flags = flags_and_scores.flags

            # Calculate percentage of anomaly flags in given range
            num_anomalies_detected = 0
            for flag in actual_flags[start : end + 1]:
                if flag == "anomaly_higher_confidence":
                    num_anomalies_detected += 1

            result = (
                "anomaly"
                if (num_anomalies_detected / (end - start + 1)) >= threshold
                else "noanomaly"
            )

            assert result == expected_type

    def test_stream_score(self):

        test_ts_mp_mulipliers = [1000, -1000, 1]
        expected_flags = ["anomaly_higher_confidence", "anomaly_higher_confidence", "none"]

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        timeseries = loaded_synthetic_data.timeseries
        timestamps = loaded_synthetic_data.timestamps
        mp_dists = loaded_synthetic_data.mp_dists
        window_sizes = loaded_synthetic_data.window_sizes

        for ts_baseline, ts_timestamps, mp_dist_baseline, window_size in zip(
            timeseries, timestamps, mp_dists, window_sizes
        ):
            ad_config = AnomalyDetectionConfig(
                time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
            )

            for i, multiplier in enumerate(test_ts_mp_mulipliers):
                test_ts_val = ts_baseline[-1] * multiplier
                test_mp_dist = mp_dist_baseline[-1] * abs(multiplier)

                flags_and_scores = self.scorer.stream_score(
                    test_ts_val,
                    ts_timestamps[-1],
                    test_mp_dist,
                    ts_baseline,
                    ts_timestamps,
                    mp_dist_baseline,
                    ad_config,
                    window_size,
                )
                assert flags_and_scores is not None
                actual_flags = flags_and_scores.flags

                assert actual_flags[0] == expected_flags[i]

    def test_failed_case(self):
        class DummyScorer(MPScorer):
            def batch_score(self, *args, **kwargs):
                return None

            def stream_score(self, *args, **kwargs):
                return None

        scorer = MPCascadingScorer(scorers=[DummyScorer(), DummyScorer()])
        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )

        flags_and_scores = scorer.batch_score(
            np.arange(1.0, 10),
            np.arange(1.0, 10),
            np.arange(1.0, 10),
            ad_config=ad_config,
            window_size=3,
        )
        assert flags_and_scores is None

        flags_and_scores = scorer.stream_score(
            np.arange(1.0, 10),
            np.arange(1.0, 10),
            np.arange(1.0, 10),
            np.arange(1.0, 10),
            np.arange(1.0, 3),
            np.arange(1.0, 3),
            ad_config=ad_config,
            window_size=3,
        )
        assert flags_and_scores is None


class TestLowVarianceScorer(unittest.TestCase):
    def test_low_variance_scorer(self):
        scorer = LowVarianceScorer(scaling_factors={"low": 5})
        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )

        with self.assertRaises(ClientError, msg="Invalid sensitivity: invalid"):
            scorer.batch_score(
                np.array([1.0] * 9),
                np.arange(1.0, 10.0),
                np.arange(1.0, 10.0),
                ad_config=ad_config,
                window_size=3,
            )

        with self.assertRaises(ClientError, msg="Invalid sensitivity: invalid"):
            scorer.stream_score(
                streamed_value=1.0,
                streamed_timestamp=1.0,
                streamed_mp_dist=1.0,
                history_values=np.array([1.0] * 9),
                history_timestamps=np.arange(1.0, 10.0),
                history_mp_dist=np.arange(1.0, 10.0),
                ad_config=ad_config,
                window_size=3,
            )

    def test_to_flag_and_score(self):
        scorer = LowVarianceScorer()
        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )

        flag, _, _ = scorer._to_flag_and_score(30, 5, ad_config)
        self.assertEqual(flag, "anomaly_higher_confidence")

        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="up", expected_seasonality="auto"
        )

        flag, _, _ = scorer._to_flag_and_score(30, 5, ad_config)
        self.assertEqual(flag, "anomaly_higher_confidence")

        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="down", expected_seasonality="auto"
        )

        flag, _, _ = scorer._to_flag_and_score(30, 5, ad_config)
        self.assertEqual(flag, "none")

        flag, _, _ = scorer._to_flag_and_score(-100, 5, ad_config)
        self.assertEqual(flag, "anomaly_higher_confidence")


class TestMPIQRScorer(unittest.TestCase):
    @patch("seer.anomaly_detection.detectors.location_detectors.ProphetLocationDetector.detect")
    def test_adjust_flag_for_non_anomalous_case(self, mock_location_detector):
        scorer = MPIQRScorer()

        mock_location_detector.return_value = PointLocation.DOWN

        # if original flag is "none". then location detector should not be called.
        mp_based_flag = "none"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag,
            "up",
            1.0,
            1.0,
            np.array([1.0] * 9),
            np.arange(1.0, 10.0),
        )
        mock_location_detector.assert_not_called()
        self.assertEqual(flag, mp_based_flag)

        mp_based_flag = "none"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag,
            "down",
            1.0,
            1.0,
            np.array([1.0] * 9),
            np.arange(1.0, 10.0),
        )
        mock_location_detector.assert_not_called()
        self.assertEqual(flag, mp_based_flag)

        mp_based_flag = "none"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag,
            "both",
            1.0,
            1.0,
            np.array([1.0] * 9),
            np.arange(1.0, 10.0),
        )
        mock_location_detector.assert_not_called()
        self.assertEqual(flag, mp_based_flag)

    @patch("seer.anomaly_detection.detectors.location_detectors.ProphetLocationDetector.detect")
    def test_adjust_flag_for_detecting_both_directions(self, mock_location_detector):
        scorer = MPIQRScorer()

        mock_location_detector.return_value = PointLocation.DOWN

        # if original flag is "none". then location detector should not be called.
        mp_based_flag = "none"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag,
            "both",
            1.0,
            1.0,
            np.array([1.0] * 9),
            np.arange(1.0, 10.0),
        )
        mock_location_detector.assert_not_called()
        self.assertEqual(flag, mp_based_flag)

        mp_based_flag = "anomaly_higher_confidence"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag,
            "both",
            1.0,
            1.0,
            np.array([1.0] * 9),
            np.arange(1.0, 10.0),
        )
        mock_location_detector.assert_not_called()
        self.assertEqual(flag, mp_based_flag)

        mp_based_flag = "anomaly_lower_confidence"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag,
            "both",
            1.0,
            1.0,
            np.array([1.0] * 9),
            np.arange(1.0, 10.0),
        )
        # self.assertEqual(mock_location_detector.call_count, 2)
        mock_location_detector.assert_not_called()
        self.assertEqual(flag, mp_based_flag)

    @patch("seer.anomaly_detection.detectors.location_detectors.ProphetLocationDetector.detect")
    def test_adjust_flag_for_anomalous_case(self, mock_location_detector):
        scorer = MPIQRScorer()
        combos = [
            {
                "mp_based_flag": "anomaly_higher_confidence",
                "direction": "up",
                "location": PointLocation.UP,
                "expected_flag": "anomaly_higher_confidence",
                "name": "up_up",
            },
            {
                "mp_based_flag": "anomaly_higher_confidence",
                "direction": "up",
                "location": PointLocation.DOWN,
                "expected_flag": "none",
                "name": "up_down",
            },
            {
                "mp_based_flag": "anomaly_higher_confidence",
                "direction": "up",
                "location": PointLocation.NONE,
                "expected_flag": "none",
                "name": "up_none",
            },
            {
                "mp_based_flag": "anomaly_higher_confidence",
                "direction": "down",
                "location": PointLocation.DOWN,
                "expected_flag": "anomaly_higher_confidence",
                "name": "down_down",
            },
            {
                "mp_based_flag": "anomaly_higher_confidence",
                "direction": "down",
                "location": PointLocation.UP,
                "expected_flag": "none",
                "name": "down_up",
            },
            {
                "mp_based_flag": "anomaly_higher_confidence",
                "direction": "up",
                "location": PointLocation.NONE,
                "expected_flag": "none",
                "name": "up_none",
            },
            {
                "mp_based_flag": "anomaly_higher_confidence",
                "direction": None,
                "location": PointLocation.NONE,
                "expected_flag": "anomaly_higher_confidence",
                "name": "up_none_failed_direction_detection",
            },
        ]

        for i, combo in enumerate(combos):
            mp_based_flag = combo["mp_based_flag"]
            direction = combo["direction"]
            location = combo["location"]
            expected_flag = combo["expected_flag"]

            mock_location_detector.return_value = location

            flag = scorer._adjust_flag_for_direction(
                mp_based_flag, direction, 1.0, 1.0, np.array([1.0] * 9), np.arange(1.0, 10.0)
            )
            self.assertEqual(mock_location_detector.call_count, i + 1)
            self.assertEqual(flag, expected_flag, msg=combo["name"])

    @patch("seer.anomaly_detection.detectors.location_detectors.ProphetLocationDetector.detect")
    def test_adjust_flag_for_anomalous_case_when_location_inbwetween(self, mock_location_detector):
        scorer = MPIQRScorer()

        mock_location_detector.return_value = PointLocation.NONE

        mp_based_flag = "anomaly_higher_confidence"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag, "up", 1.0, 1.0, np.array([1.0] * 9), np.arange(1.0, 10.0)
        )
        mock_location_detector.assert_called_once()
        self.assertEqual(flag, "none")

    @patch("seer.anomaly_detection.detectors.location_detectors.ProphetLocationDetector.detect")
    def test_adjust_flag_for_anomalous_case_when_location_fails(self, mock_location_detector):
        scorer = MPIQRScorer()

        mock_location_detector.return_value = None

        mp_based_flag = "anomaly_higher_confidence"
        flag = scorer._adjust_flag_for_direction(
            mp_based_flag, "up", 1.0, 1.0, np.array([1.0] * 9), np.arange(1.0, 10.0)
        )
        mock_location_detector.assert_called_once()
        self.assertEqual(flag, "anomaly_higher_confidence")
