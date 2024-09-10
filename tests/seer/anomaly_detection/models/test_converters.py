import unittest

import numpy as np

from seer.anomaly_detection.models import TimeSeries
from seer.anomaly_detection.models.converters import convert_external_ts_to_internal
from seer.anomaly_detection.models.external import TimeSeriesPoint


class TestConverters(unittest.TestCase):

    def test_convert_external_ts_to_internal(self):
        external_ts = []

        external_ts.append(TimeSeriesPoint(timestamp=1, value=1))
        external_ts.append(TimeSeriesPoint(timestamp=5, value=2))

        converted_ts = convert_external_ts_to_internal(external_ts)

        assert isinstance(converted_ts, TimeSeries)
        assert isinstance(converted_ts.values, np.ndarray)
        assert isinstance(converted_ts.timestamps, np.ndarray)
        assert len(converted_ts.values) == 2
        assert len(converted_ts.timestamps) == 2
