import json
import os

import numpy as np

from seer.anomaly_detection.models import TimeSeriesPoint


# Returns timeseries and mp_distances as lists of numpy arrays from the synthetic data
def convert_synthetic_ts(directory: str, as_ts_datatype: bool):

    timeseries = []
    mp_dists = []
    window_sizes = []

    # Load in time series JSON files in test_data
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            if not os.path.isfile(f):
                raise Exception("Path provided is not a valid file")
            if filename.split(".")[-1] != "json":
                raise Exception("File is not a JSON file")

            file_params = filename.split(".")[0].split("_")
            print(filename, file_params)
            window_size = int(file_params[2])

            # Load json and convert to ts and mp_dist
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

    return timeseries, mp_dists, window_sizes
