import numpy as np

from seer.anomaly_detection.models import TimeSeries
from seer.anomaly_detection.models.external import TimeSeriesPoint


def convert_external_ts_to_internal(external_ts: list[TimeSeriesPoint]) -> TimeSeries:
    values = []
    timestamps = []
    for point in external_ts:
        values.append(np.float64(point.value))
        timestamps.append(np.float64(point.timestamp))

    return TimeSeries(values=np.array(values), timestamps=np.array(timestamps))
