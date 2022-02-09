import unittest
import pandas as pd

from unittest import mock
from datetime import datetime

from docker.seer import aggregate_anomalies


class TestMain(unittest.TestCase):
    def test_aggregate_anomalies(self):
        input_dataframe = pd.DataFrame(
            {
                'ds_time': [1644367729.564212, 1644367423.377069],
                'score': [1.4, 1.6],
                'y': [5.4, 5.5],
                'yhat': [5.2, 5.3],
            })
        granularity = 300

        expected_output = [
            {'start': 1644367729, 'end': 1644368029, 'confidence': 1.4, 'received': 5.4, 'expected': 5.2, 'id': 0},
            {'start': 1644367423, 'end': 1644367723, 'confidence': 1.6, 'received': 5.5, 'expected': 5.3, 'id': 1}
        ]

        actual_output = aggregate_anomalies(input_dataframe, granularity)

        assert actual_output == expected_output


if __name__ == '__main__':
    unittest.main()
