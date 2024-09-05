import unittest

import numpy as np

from seer.anomaly_detection.detectors.mp_scorers import MPIRQScorer


class TestMPScorers(unittest.TestCase):

    def setUp(self):
        self.scorer = MPIRQScorer()

    def test_simple_batch_score(self):

        ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
        mp_dist = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)

        # TODO: sensitivity and direction are placeholders as they are not actually used in scoring yet
        sensitivity = "low"
        direction = "up"
        window_size = 2

        actual_scores, actual_flags = self.scorer.batch_score(
            ts, mp_dist, sensitivity, direction, window_size
        )

        expected_scores = [val - 16.2 for val in mp_dist]
        expected_flags = ["none"] * 9

        self.assertListEqual(actual_scores, expected_scores)
        self.assertListEqual(actual_flags, expected_flags)

    def test_batch_score_with_anomalies(self):

        ts = np.array([1, 2, 3, 4, 100, 6, 7, 8, 9], dtype=np.float64)
        mp_dist = np.array([1, 2, 3, 4, 100, 6, 7, 8, 9], dtype=np.float64)

        # TODO: sensitivity and direction are placeholders as they are not actually used in scoring yet
        sensitivity = "low"
        direction = "up"
        window_size = 2

        actual_scores, actual_flags = self.scorer.batch_score(
            ts, mp_dist, sensitivity, direction, window_size
        )

        expected_scores = [val - 18.7 for val in mp_dist]
        expected_flags = ["none"] * 9
        expected_flags[4] = "anomaly_higher_confidence"

        np.testing.assert_array_almost_equal(
            actual_scores, expected_scores
        )  # using almost equals due to rounding errors due to float
        self.assertListEqual(actual_flags, expected_flags)

    def test_stream_score(self):
        pass

        # TODO: Similar to batch_scores but output is a single value for score and flag
