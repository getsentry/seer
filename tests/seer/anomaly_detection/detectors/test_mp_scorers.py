import unittest

from seer.anomaly_detection.detectors.mp_scorers import MPCascadingScorer
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestMPScorers(unittest.TestCase):

    def setUp(self):
        self.scorer = MPCascadingScorer()

    def test_batch_score_synthetic_data(self):

        # TODO: sensitivity and direction are placeholders as they are not actually used in scoring yet
        sensitivity = ""
        direction = ""

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

    def test_stream_score(self):

        test_ts_mp_mulipliers = [1000, -1000, 1]
        expected_flags = ["anomaly_higher_confidence", "anomaly_higher_confidence", "none"]

        timeseries, mp_dists, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        for ts_baseline, mp_dist_baseline, window_size in zip(timeseries, mp_dists, window_sizes):
            sensitivity, direction = "", ""  # TODO: Placeholders as values are not used

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
