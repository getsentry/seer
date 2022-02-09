import unittest
import pandas as pd

from unittest import mock

from docker.prophet_detector import ProphetDetector
from docker.prophet_detector import ProphetParams


class TestProphetDetector(unittest.TestCase):
    def test_scale_scores(self):
        prophet_params = ProphetParams()
        prophet_detector = ProphetDetector(prophet_params)

        input_dataframe = pd.DataFrame({
            'ds': [1644367729, 1644367423],
            'y': [5.4, 5.5],
            'yhat': [5.3, 5.3],
            'yhat_upper': [5.3, 5.3],
            'yhat_lower': [5.3, 5.3],
        })

        raw_dataframe = pd.DataFrame({
            'time': [1644367729, 1644367423],

        })

        processed_dataframe = prophet_detector.pre_process_data(input_dataframe, 1644367729, 1644367423)

        actual_output = prophet_detector.scale_scores(input_dataframe)
        print(actual_output)

        assert 1 == 1
