"""
Trend Detection Logic:

- Get the most recent change point (as long as its 5 datapoints from the left and 1 from the right)
    - otherwise if the change points are on the edges use midpoint analysis
- Calculate the trend percentage before/after the breakpoint
- Use Welch’s T-test to determine if there is a significant difference in means
    - Welch’s t-test: finds significant in means between two sample groups with unequal variances and different sample sizes
- If p-value > 0.01 and trend percentage > 5%, then the trend is surfaced

"""

import pandas as pd
import numpy as np
import scipy
import datetime

from seer.trend_detection.detectors.cusum_detection import CUSUMDetector


def find_trends(
    txns_data,
    sort_function,
    allow_midpoint,
    min_pct_change,
    min_change,
    validate_tail_hours,
    pval=0.01,
):
    trend_percentage_list = []

    # defined outside for loop so error won't throw for empty data
    transaction_list = txns_data.keys()

    for txn in transaction_list:
        # data without zero-filling
        timestamps = []
        metrics = []

        # get all the non-zero data
        ts_data = txns_data[txn]["data"]

        for i in range(len(ts_data)):
            metric = ts_data[i][1][0]["count"]

            if metric != 0:
                timestamps.append(ts_data[i][0])
                metrics.append(metric)

        # snuba query limit was hit, and we won't have complete data for this transaction so disregard this txn
        if None in metrics:
            continue

        # extract all zero filled data
        timestamps_zero_filled = [ts_data[x][0] for x in range(len(ts_data))]
        metrics_zero_filled = [ts_data[x][1][0]["count"] for x in range(len(ts_data))]

        req_start = int(txns_data[txn]["request_start"])
        req_end = int(txns_data[txn]["request_end"])

        # don't include transaction if there are less than three datapoints in non zero data OR
        # don't include transaction if there is no more data within request time period
        if len(metrics) < 3 or req_start > timestamps[-1]:
            continue

        try:
            # grab the index of the request start time
            req_start_index = next(i for i, v in enumerate(timestamps) if v > req_start)
        except StopIteration:
            # After removing the zerofilled entries, it's possible that all
            # timestamps fall before the request start. When this happens, there
            # is no trend to be found.
            continue

        # convert to pandas timestamps for magnitude compare method in cusum detection
        timestamps_pandas = [
            pd.Timestamp(datetime.datetime.fromtimestamp(x)) for x in timestamps
        ]
        timestamps_zerofilled_pandas = [
            pd.Timestamp(datetime.datetime.fromtimestamp(x))
            for x in timestamps_zero_filled
        ]

        timeseries = pd.DataFrame({"time": timestamps_pandas, "y": metrics})

        timeseries_zerofilled = pd.DataFrame(
            {"time": timestamps_zerofilled_pandas, "y": metrics_zero_filled}
        )

        change_points = CUSUMDetector(timeseries, timeseries_zerofilled).detector()

        # get number of breakpoints in second half of timeseries
        num_breakpoints = len(change_points)

        # sort change points by start time to get most recent one
        change_points.sort(key=lambda x: x.start_time)

        # if breakpoints are detected, get most recent changepoint
        if num_breakpoints != 0:
            change_point = change_points[-1].start_time
            # convert back to datetime timestamp
            change_point = int(datetime.datetime.timestamp(change_point))
            change_index = timestamps.index(change_point)

        # if breakpoint is in the very beginning or no breakpoints are detected, use midpoint analysis instead
        if num_breakpoints == 0 or change_index <= 5:
            # check the midpoint boolean - don't get midpoint of the request period if this boolean is false, midpoint should only be used for trends
            if not allow_midpoint:
                continue

            change_point = (req_start + req_end) // 2

        first_half = [
            metrics[i]
            for i in range(len(metrics))
            if timestamps[i] < change_point and timestamps[i] >= req_start
        ]
        second_half = [
            metrics[i]
            for i in range(len(metrics))
            if timestamps[i] >= change_point and timestamps[i] <= req_end
        ]

        # if either of the halves don't have any data to compare to then move on to the next txn
        if len(first_half) == 0 or len(second_half) == 0:
            continue

        mu0 = np.average(first_half)
        mu1 = np.average(second_half)

        # calculate t-value between both groups
        scipy_t_test = scipy.stats.ttest_ind(first_half, second_half, equal_var=False)

        if mu0 == 0:
            trend_percentage = mu1
        else:
            trend_percentage = mu1 / mu0

        txn_names = txn.split(",")
        output_dict = {
            "project": txn_names[0],
            "transaction": txn_names[1],
            "aggregate_range_1": mu0,
            "aggregate_range_2": mu1,
            "unweighted_t_value": scipy_t_test.statistic,
            "unweighted_p_value": round(scipy_t_test.pvalue, 10),
            "trend_percentage": trend_percentage,
            "absolute_percentage_change": abs(trend_percentage),
            "trend_difference": mu1 - mu0,
            "breakpoint": change_point,
            "request_start": req_start,
            "request_end": req_end,
            "data_start": int(txns_data[txn]["data_start"]),
            "data_end": int(txns_data[txn]["data_end"]),
        }

        # TREND LOGIC:
        #  1. p-value of t-test is less than passed in threshold (default = 0.01)
        #  2. trend percentage is greater than passed in threshold (default = 10%)
        #  3. last validate_tail_hours hours are also greater than threshold

        if validate_tail_hours > 0:
            validation_start = req_end - validate_tail_hours * 60 * 60

            # Filter out the data for the last 24 hours
            validation_data = [
                metric
                for timestamp, metric in zip(timestamps, metrics)
                if max(validation_start, change_point) <= timestamp <= req_end
            ]

            # Calculate the trend percentage and change for the last validate_tail_hours
            mu_validation = np.average(validation_data)
            trend_percentage_validation = (
                mu_validation / mu0 if mu0 != 0 else mu_validation
            )
            trend_change_validation = mu_validation - mu0
        else:
            trend_percentage_validation = None
            trend_change_validation = None

        # most improved - get only negatively significant trending txns
        if (
            (sort_function == "trend_percentage()" or sort_function == "")
            and mu1 + min_change <= mu0
            and scipy_t_test.pvalue < pval
            and abs(trend_percentage - 1) > min_pct_change
            and (
                trend_percentage_validation is None
                or abs(trend_percentage_validation - 1) > min_pct_change
            )
            and (
                trend_change_validation is None
                or abs(trend_change_validation) > min_change
            )
        ):
            output_dict["change"] = "improvement"
            trend_percentage_list.append((trend_percentage, output_dict))

        # if most regressed - get only positively significant txns
        elif (
            (sort_function == "-trend_percentage()" or sort_function == "")
            and mu0 + min_change <= mu1
            and scipy_t_test.pvalue < pval
            and trend_percentage - 1 > min_pct_change
            and (
                trend_percentage_validation is None
                or trend_percentage_validation - 1 > min_pct_change
            )
            and (
                trend_change_validation is None or trend_change_validation > min_change
            )
        ):
            output_dict["change"] = "regression"
            trend_percentage_list.append((trend_percentage, output_dict))

    return trend_percentage_list
