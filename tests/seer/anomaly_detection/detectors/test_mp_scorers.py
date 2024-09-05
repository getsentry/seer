import json
import os
import unittest

import numpy as np

from seer.anomaly_detection.detectors.mp_scorers import MPIRQScorer


class TestMPScorers(unittest.TestCase):

    def setUp(self):
        self.scorer = MPIRQScorer()

    # def test_simple_batch_score(self):

    #     ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    #     mp_dist = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)

    #     # TODO: sensitivity and direction are placeholders as they are not actually used in scoring yet
    #     sensitivity = "low"
    #     direction = "up"
    #     window_size = 2

    #     actual_scores, actual_flags = self.scorer.batch_score(
    #         ts, mp_dist, sensitivity, direction, window_size
    #     )

    #     expected_scores = [val - 16.2 for val in mp_dist]
    #     expected_flags = ["none"] * 9

    #     self.assertListEqual(actual_scores, expected_scores)
    #     self.assertListEqual(actual_flags, expected_flags)

    # def test_batch_score_with_anomalies(self):

    #     ts = np.array([1, 2, 3, 4, 100, 6, 7, 8, 9], dtype=np.float64)
    #     mp_dist = np.array([1, 2, 3, 4, 100, 6, 7, 8, 9], dtype=np.float64)

    #     # TODO: sensitivity and direction are placeholders as they are not actually used in scoring yet
    #     sensitivity = "low"
    #     direction = "up"
    #     window_size = 2

    #     actual_scores, actual_flags = self.scorer.batch_score(
    #         ts, mp_dist, sensitivity, direction, window_size
    #     )

    #     expected_scores = [val - 18.7 for val in mp_dist]
    #     expected_flags = ["none"] * 9
    #     expected_flags[4] = "anomaly_higher_confidence"

    #     np.testing.assert_array_almost_equal(
    #         actual_scores, expected_scores
    #     )  # using almost equals due to rounding errors due to float
    #     self.assertListEqual(actual_flags, expected_flags)

    def test_batch_score_synthetic_data(self):

        def is_anomaly_detected(filename, threshold, expected_type, window_size, start, end):

            if not os.path.isfile(filename):
                raise Exception("Filename is not a valid file")

            # Load json and convert to ts and mp_dist
            with open(filename) as f:

                data = json.load(f)
                data = data["ts"]

                ts = np.array([point["value"] for point in data], dtype=np.float64)
                mp_dist = np.array([point["mp_dist"] for point in data], dtype=np.float64)

                # TODO: sensitivity and direction are placeholders as they are not actually used in scoring yet
                sensitivity = ""
                direction = ""

                actual_scores, actual_flags = self.scorer.batch_score(
                    ts, mp_dist, sensitivity, direction, window_size
                )

                # Calculate percentage of anomaly flags in given range
                num_anomalies_detected = 0
                for flag in actual_flags[start : end + 1]:
                    if flag == "anomaly_higher_confidence":
                        num_anomalies_detected += 1

                return (
                    "anomaly"
                    if (num_anomalies_detected / (end - start + 1)) >= threshold
                    else "nonanomaly"
                )

        actual_results = []
        expected_results = []

        # Check time series JSON files in test_data
        dir = "tests/seer/anomaly_detection/detectors/test_data/synthetic_series"
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)

            if os.path.isfile(f):
                file_params = filename.split(".")[0].split("_")
                expected_type, window_size, start, end = (
                    file_params[1],
                    int(file_params[2]),
                    int(file_params[3]),
                    int(file_params[4]),
                )
                actual_results.append(
                    is_anomaly_detected(f, 0.1, expected_type, window_size, start, end)
                )
                expected_results.append(expected_type)

        self.assertListEqual(actual_results, expected_results)

    def test_stream_score(self):
        pass

        # TODO: Similar to batch_scores but output is a single value for score and flag
