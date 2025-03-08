import json
import os
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from seer.anomaly_detection.models import TimeSeriesPoint


class LoadedSyntheticData(BaseModel):
    timeseries: List[np.ndarray] | List[List[TimeSeriesPoint]]
    timestamps: List[np.ndarray]
    mp_dists: List[np.ndarray]
    window_sizes: List[int]
    expected_types: Optional[List[str]] = Field(None)
    anomaly_starts: Optional[List[int]] = Field(None)
    anomaly_ends: Optional[List[int]] = Field(None)
    filenames: Optional[List[str]] = Field(None)
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


def test_data_with_cycles(num_days: int = 29, num_anomalous: int = 5) -> pd.DataFrame:
    """
    Creates a time series with daily cycles and adds anomalies to the last num_anomalous points.

    Parameters:
    num_anomalous: int
        Number of anomalous points to add to the time series.

    Returns:
    pd.DataFrame: A DataFrame with the time series data.
    """
    date_range = pd.date_range(start="2023-01-01", periods=num_days, freq="D")

    # Generate daily cycle values
    daily_cycle = np.sin(2 * np.pi * np.arange(24) / 24)

    # Create a time series with daily cycles
    data = []
    for date in date_range:
        for hour in range(24):
            value = daily_cycle[hour] + np.random.normal(0, 0.2)  # Add some noise
            data.append({"timestamp": date + timedelta(hours=hour), "value": value})

    for i in range(1, num_anomalous):
        data[-i]["value"] = data[-i]["value"] + 1.5  # add anomaly for the last num_anomalous points
    df = pd.DataFrame(data)
    return df


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
    timestamps = []
    mp_dists = []
    window_sizes = []
    anomaly_starts = []
    anomaly_ends = []
    expected_types = []
    filenames = []

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

                values = None
                num_rows = len(data)
                gen_timestamps = pd.date_range(
                    start="2024-01-01", periods=num_rows, freq="15min", tz="UTC", unit="s"
                ).values.astype(np.int64)
                if as_ts_datatype:
                    values = [
                        TimeSeriesPoint(timestamp=float(ts), value=point["value"])
                        for ts, point in zip(gen_timestamps, data)
                    ]
                else:
                    values = np.array([point["value"] for point in data], dtype=np.float64)

                ts_timestamps = np.array(gen_timestamps, dtype=np.float64)
                mp_dist = np.array([point["mp_dist"] for point in data], dtype=np.float64)

                timeseries.append(values)
                timestamps.append(ts_timestamps)
                mp_dists.append(mp_dist)
                window_sizes.append(window_size)
                anomaly_starts.append(start)
                anomaly_ends.append(end)
                expected_types.append(expected_type)
                filenames.append(filename)
    if include_anomaly_range:
        return LoadedSyntheticData(
            expected_types=expected_types,
            timeseries=timeseries,
            timestamps=timestamps,
            mp_dists=mp_dists,
            window_sizes=window_sizes,
            anomaly_starts=anomaly_starts,
            anomaly_ends=anomaly_ends,
            filenames=filenames,
        )
    return LoadedSyntheticData(
        timeseries=timeseries,
        timestamps=timestamps,
        mp_dists=mp_dists,
        window_sizes=window_sizes,
        filenames=filenames,
    )
