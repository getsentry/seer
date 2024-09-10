import json
import os

import numpy as np

from seer.anomaly_detection.models import TimeSeriesPoint


# Returns timeseries and mp_distances as lists of numpy arrays from the synthetic data
def convert_synthetic_ts(directory: str, as_ts_datatype: bool, include_anomaly_range: bool = False):
    """
    Main entry point for storing time series data for an alert. Synthetic data must be a JSON
    file with the appropriate key-value pairs (timestamp: int, value: float, mp_dist: float) and
    with name in the format '<id>_<expected_anomaly>_<window_size>_<anomaly_start>_<anomaly_end>.json'

    Parameters:
    directory: string
        Target directory where synthetic data is to be retrieved and parsed from.

    as_ts_datatype: bool
        Flag for if timeseries should be returned as a List[TimeSeriesPoint] rather than np.array

    include_anomaly_range: bool
        Flag for if anomaly range [start, end] should be returned.

    Returns:
    Tuple of lists for each attribute depending on the flag
    """

    timeseries = []
    mp_dists = []
    window_sizes = []
    anomaly_starts = []
    anomaly_ends = []
    expected_types = []

    # Load in time series JSON files in test_data
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            if not os.path.isfile(f):
                raise Exception("Path provided is not a valid file")
            if filename.split(".")[-1] != "json":
                raise Exception("File is not a JSON file")

            # Parse file name
            file_params = filename.split(".")[0].split("_")
            expected_type, window_size, start, end = (
                file_params[1],
                int(file_params[2]),
                int(file_params[3]),
                int(file_params[4]),
            )

            # Load json to convert to appropriate outputs and append to respective lists
            with open(f) as file:

                data = json.load(file)
                data = data["ts"]

                ts = None
                if as_ts_datatype:
                    ts = [
                        TimeSeriesPoint(timestamp=point["timestamp"], value=point["value"])
                        for point in data
                    ]
                else:
                    ts = np.array([point["value"] for point in data], dtype=np.float64)

                mp_dist = np.array([point["mp_dist"] for point in data], dtype=np.float64)

                timeseries.append(ts)
                mp_dists.append(mp_dist)
                window_sizes.append(window_size)
                anomaly_starts.append(start)
                anomaly_ends.append(end)
                expected_types.append(expected_type)

    if include_anomaly_range:
        return expected_types, timeseries, mp_dists, window_sizes, anomaly_starts, anomaly_ends
    return timeseries, mp_dists, window_sizes
