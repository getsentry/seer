import unittest

import numpy as np

from seer.anomaly_detection.detectors.mp_scorers import MPCascadingScorer, MPScorer
from seer.exceptions import ClientError
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestMPScorers(unittest.TestCase):

    def setUp(self):
        self.scorer = MPCascadingScorer()

    def test_batch_score_synthetic_data(self):

        # TODO: sensitivity and direction are placeholders as they are not actually used in scoring yet
        sensitivity = "high"
        direction = "both"

        expected_types, timeseries, mp_dists, window_sizes, window_starts, window_ends = (
            convert_synthetic_ts(
                "tests/seer/anomaly_detection/test_data/synthetic_series",
                as_ts_datatype=False,
                include_anomaly_range=True,
            )
        )

        threshold = 0.1

        for expected_type, ts, mp_dist, window_size, start, end in zip(
            expected_types, timeseries, mp_dists, window_sizes, window_starts, window_ends
        ):

            flags_and_scores = self.scorer.batch_score(
                ts, mp_dist, sensitivity, direction, window_size
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

    def test_batch_score_invalid_sensitivity(self):

        timeseries, mp_dists, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )
        ts_baseline, mp_dist_baseline, window_size = timeseries[0], mp_dists[0], window_sizes[0]
        sensitivity, direction = "invalid", "both"

        with self.assertRaises(ClientError, msg="Invalid sensitivity: invalid"):
            self.scorer.batch_score(
                ts_baseline,
                mp_dist_baseline,
                sensitivity,
                direction,
                window_size,
            )

    def test_batch_score_invalid_sensitivity_flat_ts(self):

        ts_baseline = np.ones(200)
        mp_dist_baseline = np.ones(200)
        window_size = 3

        sensitivity, direction = "invalid", "both"

        with self.assertRaises(ClientError, msg="Invalid sensitivity: invalid"):
            self.scorer.batch_score(
                ts_baseline,
                mp_dist_baseline,
                sensitivity,
                direction,
                window_size,
            )

    def test_stream_score(self):

        test_ts_mp_mulipliers = [1000, -1000, 1]
        expected_flags = ["anomaly_higher_confidence", "anomaly_higher_confidence", "none"]

        timeseries, mp_dists, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        for ts_baseline, mp_dist_baseline, window_size in zip(timeseries, mp_dists, window_sizes):
            sensitivity, direction = "high", "both"

            for i, multiplier in enumerate(test_ts_mp_mulipliers):
                test_ts_val = ts_baseline[-1] * multiplier
                test_mp_dist = mp_dist_baseline[-1] * abs(multiplier)

                flags_and_scores = self.scorer.stream_score(
                    test_ts_val,
                    test_mp_dist,
                    ts_baseline,
                    mp_dist_baseline,
                    sensitivity,
                    direction,
                    window_size,
                )
                assert flags_and_scores is not None
                actual_flags = flags_and_scores.flags

                assert actual_flags[0] == expected_flags[i]

    def test_stream_score_invalid_sensitivity(self):

        timeseries, mp_dists, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )
        ts_baseline, mp_dist_baseline, window_size = timeseries[0], mp_dists[0], window_sizes[0]
        sensitivity, direction = "invalid", "both"

        with self.assertRaises(ClientError, msg="Invalid sensitivity: invalid"):
            self.scorer.stream_score(
                timeseries[-1],
                mp_dists[-1],
                ts_baseline,
                mp_dist_baseline,
                sensitivity,
                direction,
                window_size,
            )

    def test_stream_score_invalid_sensitivity_flat_ts(self):

        ts_baseline = np.ones(200)
        mp_dist_baseline = np.ones(200)
        window_size = 3

        sensitivity, direction = "invalid", "both"

        with self.assertRaises(ClientError, msg="Invalid sensitivity: invalid"):
            self.scorer.stream_score(
                ts_baseline[-1],
                mp_dist_baseline[-1],
                ts_baseline,
                mp_dist_baseline,
                sensitivity,
                direction,
                window_size,
            )

    def test_cascading_scorer_failed_case(self):
        class DummyScorer(MPScorer):
            def batch_score(self, *args, **kwargs):
                return None

            def stream_score(self, *args, **kwargs):
                return None

        scorer = MPCascadingScorer(scorers=[DummyScorer(), DummyScorer()])

        flags_and_scores = scorer.batch_score(
            np.arange(1.0, 10),
            np.arange(1.0, 10),
            sensitivity="high",
            direction="both",
            window_size=3,
        )
        assert flags_and_scores is None

        flags_and_scores = scorer.stream_score(
            np.arange(1.0, 10),
            np.arange(1.0, 10),
            np.arange(1.0, 3),
            np.arange(1.0, 3),
            sensitivity="high",
            direction="both",
            window_size=3,
        )
        assert flags_and_scores is None
