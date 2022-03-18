import unittest
import json
import pandas as pd

from unittest import mock

from seer.seer import aggregate_anomalies, process_output, predict


class TestSeer(unittest.TestCase):
    def test_aggregate_anomalies(self):
        input_dataframe = pd.DataFrame(
            {
                "ds_time": [1644367729.564212, 1644367423.377069],
                "score": [1, 2],
                "y": [5.4, 5.5],
                "yhat": [5.2, 5.3],
            }
        )
        granularity = 300
        expected_output = [
            {
                "start": 1644367729,
                "end": 1644367723,
                "confidence": "high",
                "received": 10.9,
                "expected": 10.5,
                "id": 0,
            },
        ]

        actual_output = aggregate_anomalies(input_dataframe, granularity)

        assert actual_output == expected_output

    #@mock.patch("seer.seer.flask.request")
    def test_empty_dataset(self):
        input_data = {"data": [],}

        expected_output = {
                "y": {"data": []},
                "yhat_upper": {"data": []},
                "yhat_lower": {"data": []},
                "anomalies": [],
            }

        actual_output = predict(input_data)

        assert actual_output == expected_output

    @mock.patch("seer.seer.aggregate_anomalies")
    def test_process_output(self, mock_aggregate_anomalies):
        mock_aggregate_anomalies.return_value = [
            {
                "start": 1644367729,
                "end": 1644368029,
                "confidence": 1.4,
                "received": 5.4,
                "expected": 5.2,
                "id": 0,
            },
            {
                "start": 1644367423,
                "end": 1644367723,
                "confidence": 1.6,
                "received": 5.5,
                "expected": 5.3,
                "id": 1,
            },
        ]
        input_dataframe = pd.DataFrame(
            {
                "ds": [1644367729, 1644367423],
                "y": [5.4, 5.5],
                "yhat": [5.3, 5.3],
                "yhat_upper": [5.3, 5.3],
                "yhat_lower": [5.3, 5.3],
                "anomalies": [5.2, 5.1],
            }
        )
        expected_output = {
            "y": {
                "data": [(1, [{"count": 5.4}]), (1, [{"count": 5.5}])],
                "start": 1,
                "end": 1,
            },
            "yhat_upper": {
                "data": [(1, [{"count": 5.3}]), (1, [{"count": 5.3}])],
                "start": 1,
                "end": 1,
            },
            "yhat_lower": {
                "data": [(1, [{"count": 5.3}]), (1, [{"count": 5.3}])],
                "start": 1,
                "end": 1,
            },
            "anomalies": [
                {
                    "start": 1644367729,
                    "end": 1644368029,
                    "confidence": 1.4,
                    "received": 5.4,
                    "expected": 5.2,
                    "id": 0,
                },
                {
                    "start": 1644367423,
                    "end": 1644367723,
                    "confidence": 1.6,
                    "received": 5.5,
                    "expected": 5.3,
                    "id": 1,
                },
            ],
        }
        granularity = 300

        actual_output = process_output(input_dataframe, granularity)

        assert actual_output == expected_output


if __name__ == "__main__":
    unittest.main()
