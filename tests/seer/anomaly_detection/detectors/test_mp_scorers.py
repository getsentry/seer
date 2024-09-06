import json
import os
import unittest

import numpy as np

from seer.anomaly_detection.detectors.mp_scorers import MPIRQScorer


class TestMPScorers(unittest.TestCase):

    def setUp(self):
        self.scorer = MPIRQScorer()

    def test_batch_score_synthetic_data(self):

        def is_anomaly_detected(filename, threshold, window_size, start, end):

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
                    else "noanomaly"
                )

        actual_results = []
        expected_results = []

        # Check time series JSON files in test_data
        dir = "tests/seer/anomaly_detection/detectors/test_data/synthetic_series"
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)

            if os.path.isfile(f):
                # filename is in format expected_type, window_size, start, end separated by '_'
                file_params = filename.split(".")[0].split("_")
                expected_type, window_size, start, end = (
                    file_params[1],
                    int(file_params[2]),
                    int(file_params[3]),
                    int(file_params[4]),
                )
                actual_results.append(is_anomaly_detected(f, 0.1, window_size, start, end))
                expected_results.append(expected_type)

        self.assertListEqual(actual_results, expected_results)

    def test_stream_score(self):

        test_ts_mp_mulipliers = [1000, -1000, 1]
        expected_flags = ["anomaly_higher_confidence", "anomaly_higher_confidence", "none"]

        # Check time series JSON files in test_data
        dir = "tests/seer/anomaly_detection/detectors/test_data/synthetic_series"
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)

            if os.path.isfile(f):
                if not os.path.isfile(f):
                    raise Exception("Filename is not a valid file")

                file_params = filename.split(".")[0].split("_")
                window_size = int(file_params[2])

                # Load json and convert to ts and mp_dist
                with open(f) as file:

                    data = json.load(file)
                    data = data["ts"]

                    ts_baseline = np.array([point["value"] for point in data], dtype=np.float64)
                    mp_dist_baseline = np.array(
                        [point["mp_dist"] for point in data], dtype=np.float64
                    )

                    sensitivity, direction = "", ""  # TODO: Placeholders as values are not used

                    for i, multiplier in enumerate(test_ts_mp_mulipliers):
                        test_ts_val = ts_baseline[-1] * multiplier
                        test_mp_dist = mp_dist_baseline[-1] * abs(multiplier)

                        _, flag = self.scorer.stream_score(
                            test_ts_val,
                            test_mp_dist,
                            sensitivity,
                            direction,
                            window_size,
                            ts_baseline,
                            mp_dist_baseline,
                        )

                        self.assertEqual(flag[0], expected_flags[i])
