import unittest

import numpy as np

from seer.anomaly_detection.detectors import MPCascadingScorer, MPLowVarianceScorer, MPScorer
from seer.anomaly_detection.models import AnomalyDetectionConfig, ThresholdType
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
        filenames = loaded_synthetic_data.filenames
        threshold = 0.1

        for expected_type, ts, ts_timestamps, mp_dist, window_size, start, end, filename in zip(
            expected_types,
            timeseries,
            timestamps,
            mp_dists,
            window_sizes,
            window_starts,
            window_ends,
            filenames,
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
            assert (
                result == expected_type
            ), f"Expected for {filename}: {expected_type}, got {result}"

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
                self.assertEqual(
                    flags_and_scores.thresholds[0][0].type, ThresholdType.BOX_COX_THRESHOLD
                )

    def test_stream_score_with_thresholds(self):

        expected_flag = "anomaly_higher_confidence"

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        ts_baseline = loaded_synthetic_data.timeseries[0]
        ts_timestamps = loaded_synthetic_data.timestamps[0]
        mp_dist_baseline = loaded_synthetic_data.mp_dists[0]
        window_size = loaded_synthetic_data.window_sizes[0]

        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )

        test_ts_val = ts_baseline[-1] * 10
        test_mp_dist = mp_dist_baseline[-1] * 10

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

        assert actual_flags[0] == expected_flag
        self.assertEqual(len(flags_and_scores.thresholds), 1)
        self.assertEqual(len(flags_and_scores.thresholds[0]), 1)
        self.assertEqual(flags_and_scores.thresholds[0][0].type, ThresholdType.BOX_COX_THRESHOLD)
        self.assertLess(flags_and_scores.thresholds[0][0].lower, 0.0)
        self.assertAlmostEqual(
            flags_and_scores.thresholds[0][0].upper,
            -flags_and_scores.thresholds[0][0].lower,
            places=2,
        )

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


class TestMPLowVarianceScorer(unittest.TestCase):
    def test_low_variance_scorer(self):
        scorer = MPLowVarianceScorer(scaling_factors={"low": 5})
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
        scorer = MPLowVarianceScorer()
        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )

        flag, _, _, _ = scorer._to_flag_and_score(30, 5, ad_config)
        self.assertEqual(flag, "anomaly_higher_confidence")

        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="up", expected_seasonality="auto"
        )

        flag, _, _, _ = scorer._to_flag_and_score(30, 5, ad_config)
        self.assertEqual(flag, "anomaly_higher_confidence")

        ad_config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="down", expected_seasonality="auto"
        )

        flag, _, _, _ = scorer._to_flag_and_score(30, 5, ad_config)
        self.assertEqual(flag, "none")

        flag, _, _, _ = scorer._to_flag_and_score(-100, 5, ad_config)
        self.assertEqual(flag, "anomaly_higher_confidence")


# class TestMPIQRScorer(unittest.TestCase):
#     @patch("seer.anomaly_detection.detectors.location_detectors.ProphetLocationDetector.detect")
#     def test_abandon_batch_detection_if_time_budget_is_exceeded(self, mock_location_detector):

#         def slow_function(streamed_value, streamed_timestamp, history_values, history_timestamps):
#             time.sleep(0.05)  # Simulate a 50ms delay.
#             return None

#         mock_location_detector.side_effect = slow_function

#         scorer = MPIQRScorer()
#         mp_utils = resolve(MPUtils)
#         ws_selector = resolve(WindowSizeSelector)
#         ad_config = AnomalyDetectionConfig(
#             time_period=60, sensitivity="high", direction="up", expected_seasonality="auto"
#         )
#         algo_config = resolve(AlgoConfig)
#         df = test_data_with_cycles(num_anomalous=20)
#         window_size = ws_selector.optimal_window_size(df["value"].values)
#         mp = stumpy.stump(
#             df["value"].values,
#             m=max(3, window_size),
#             ignore_trivial=algo_config.mp_ignore_trivial,
#             normalize=False,
#         )

#         # We do not normalize the matrix profile here as normalizing during stream detection later is not straighforward.
#         mp_dist = mp_utils.get_mp_dist_from_mp(mp, pad_to_len=len(df["value"].values))
#         time_budget_ms = 100
#         with self.assertRaises(ServerError) as e:
#             scorer.batch_score(
#                 df["value"].values,
#                 df["timestamp"].values,
#                 mp_dist,
#                 ad_config=ad_config,
#                 algo_config=algo_config,
#                 window_size=window_size,
#                 time_budget_ms=time_budget_ms,
#             )
#         # Since slow func sleeps for 50 ms and timeout is 100ms, location detection should be called at least twice and upto 10 which is the batch size.
#         assert mock_location_detector.call_count >= 2
#         assert mock_location_detector.call_count <= 10
#         assert "Batch detection took too long" in str(e.exception)

#     @patch("seer.anomaly_detection.detectors.location_detectors.ProphetLocationDetector.detect")
#     def test_prophet_optimization_for_batch(self, mock_location_detector):
#         scorer = MPIQRScorer()
#         mp_utils = resolve(MPUtils)
#         ws_selector = resolve(WindowSizeSelector)
#         ad_config = AnomalyDetectionConfig(
#             time_period=60, sensitivity="high", direction="up", expected_seasonality="auto"
#         )
#         algo_config = resolve(AlgoConfig)
#         algo_config.direction_detection_num_timesteps_in_batch_mode = 2
#         df = test_data_with_cycles(num_anomalous=5)
#         window_size = ws_selector.optimal_window_size(df["value"].values)
#         # Get the matrix profile for the time series
#         mp = stumpy.stump(
#             df["value"].values,
#             m=max(3, window_size),
#             ignore_trivial=algo_config.mp_ignore_trivial,
#             normalize=False,
#         )

#         # We do not normalize the matrix profile here as normalizing during stream detection later is not straighforward.
#         mp_dist = mp_utils.get_mp_dist_from_mp(mp, pad_to_len=len(df["value"].values))

#         scorer.batch_score(
#             df["value"].values,
#             df["timestamp"].values,
#             mp_dist,
#             ad_config=ad_config,
#             algo_config=algo_config,
#             window_size=window_size,
#         )
#         assert mock_location_detector.call_count == 2

#         algo_config.direction_detection_num_timesteps_in_batch_mode = 1
#         scorer.batch_score(
#             df["value"].values,
#             df["timestamp"].values,
#             mp_dist,
#             ad_config=ad_config,
#             algo_config=algo_config,
#             window_size=window_size,
#         )
#         assert mock_location_detector.call_count == 3

#     def test_adjust_flag_for_non_anomalous_case(self):
#         scorer = MPIQRScorer()

#         # if original flag is "none". then location detector should not be called.
#         mp_based_flag = "none"
#         flag, _ = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "up",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, mp_based_flag)

#         mp_based_flag = "none"
#         flag, _ = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "down",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, mp_based_flag)

#         mp_based_flag = "none"
#         flag, _ = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "both",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, mp_based_flag)

#     def test_adjust_flag_for_detecting_both_directions(self):
#         scorer = MPIQRScorer()

#         # if original flag is "none". then location detector should not be called.
#         mp_based_flag = "none"
#         flag, _ = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "both",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, mp_based_flag)

#         mp_based_flag = "anomaly_higher_confidence"
#         flag, _ = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "both",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, mp_based_flag)

#         mp_based_flag = "anomaly_lower_confidence"
#         flag, _ = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "both",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, mp_based_flag)

#     def test_adjust_flag_for_anomalous_case(self):
#         scorer = MPIQRScorer()
#         combos = [
#             {
#                 "mp_based_flag": "anomaly_higher_confidence",
#                 "direction": "up",
#                 "location": PointLocation.UP,
#                 "expected_flag": "anomaly_higher_confidence",
#                 "name": "up_up",
#             },
#             {
#                 "mp_based_flag": "anomaly_higher_confidence",
#                 "direction": "up",
#                 "location": PointLocation.DOWN,
#                 "expected_flag": "none",
#                 "name": "up_down",
#             },
#             {
#                 "mp_based_flag": "anomaly_higher_confidence",
#                 "direction": "up",
#                 "location": PointLocation.NONE,
#                 "expected_flag": "none",
#                 "name": "up_none",
#             },
#             {
#                 "mp_based_flag": "anomaly_higher_confidence",
#                 "direction": "down",
#                 "location": PointLocation.DOWN,
#                 "expected_flag": "anomaly_higher_confidence",
#                 "name": "down_down",
#             },
#             {
#                 "mp_based_flag": "anomaly_higher_confidence",
#                 "direction": "down",
#                 "location": PointLocation.UP,
#                 "expected_flag": "none",
#                 "name": "down_up",
#             },
#             {
#                 "mp_based_flag": "anomaly_higher_confidence",
#                 "direction": "up",
#                 "location": PointLocation.NONE,
#                 "expected_flag": "none",
#                 "name": "up_none",
#             },
#             {
#                 "mp_based_flag": "anomaly_higher_confidence",
#                 "direction": None,
#                 "location": PointLocation.NONE,
#                 "expected_flag": "anomaly_higher_confidence",
#                 "name": "up_none_failed_direction_detection",
#             },
#         ]

#         for i, combo in enumerate(combos):
#             mp_based_flag = combo["mp_based_flag"]
#             direction = combo["direction"]
#             location = RelativeLocation(location=combo["location"], thresholds=[])
#             expected_flag = combo["expected_flag"]

#             flag, _ = scorer._adjust_flag_for_direction(
#                 mp_based_flag,
#                 direction,
#                 1.0,
#                 1.0,
#                 np.array([1.0] * 9),
#                 np.arange(1.0, 10.0),
#             )
#             self.assertEqual(flag, expected_flag, msg=combo["name"])

#     def test_adjust_flag_for_anomalous_case_when_location_inbwetween(self):
#         scorer = MPIQRScorer()

#         mp_based_flag = "anomaly_higher_confidence"
#         flag, _ = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "up",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, "none")

#     def test_adjust_flag_for_anomalous_case_when_location_fails(self):
#         scorer = MPIQRScorer()

#         mp_based_flag = "anomaly_higher_confidence"
#         flag, thresholds = scorer._adjust_flag_for_direction(
#             mp_based_flag,
#             "up",
#             1.0,
#             1.0,
#             np.array([1.0] * 9),
#             np.arange(1.0, 10.0),
#         )
#         self.assertEqual(flag, "anomaly_higher_confidence")
